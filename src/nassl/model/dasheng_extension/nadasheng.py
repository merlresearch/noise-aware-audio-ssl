# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (C) 2024 Xiaomi Corporation
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from Dasheng: https://github.com/RicherMans/Dasheng

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

from nassl.model.noiseaware import NoiseAwareModel
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


class NADasheng(NoiseAwareModel):
    def __init__(
        self,
        fusion_layer_cfg: dict[str, Any],
        model_id: str = "mispeech/dasheng-base",
        trainable_base: bool = True,
        eval_base: bool = True,
    ) -> None:
        super().__init__()

        self.sr = 16000
        self.trainable_base = trainable_base
        self.eval_base = eval_base

        # Load the original Dasheng model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        _wrapper = AutoModel.from_pretrained(model_id, outputdim=None, trust_remote_code=True)
        self.org_model = _wrapper.encoder
        self.embedding_dim = self.org_model.embed_dim
        for name, param in self.org_model.named_parameters():
            param.requires_grad = trainable_base

        # Create fusion layers
        num_layers = len(self.org_model.blocks)
        fusion_layer_cfg = {**fusion_layer_cfg, "embed_dim": self.embedding_dim}
        self.fusion_layers = nn.ModuleList([instantiate(fusion_layer_cfg) for _ in range(num_layers)])
        self._set_frozen_base_eval()

    def _set_frozen_base_eval(self) -> None:
        if self.trainable_base or not self.eval_base:
            return
        self.org_model.eval()

    def train(self, mode: bool = True) -> NADasheng:
        super().train(mode=mode)
        self._set_frozen_base_eval()
        return self

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """
        Args:
            x: mel spectrogram [B, 1, T, F]
        Returns:
            x: patch embeddings
        """

        # preprocessing in forward of original model
        x = self.org_model.init_bn(x) if self.org_model.init_bn is not None else x
        # Remember starting position if we pad
        padding_start = 0
        if x.shape[-1] > self.org_model.target_length:
            splits = x.split(self.org_model.target_length, -1)

            if splits[-1].shape[-1] < self.org_model.target_length:
                if self.org_model.pad_last:
                    pad = torch.zeros(*x.shape[:-1], self.org_model.target_length, device=x.device)
                    pad[..., : splits[-1].shape[-1]] = splits[-1]
                    padding_start = x.shape[-1] // self.org_model.patch_stride[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = torch.flatten(splits, 0, 1)  # spl b c f t-> (spl b) c f t
        else:
            n_splits = 1

        # forward_features in original model
        x = self.org_model.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.org_model.time_pos_embed[:, :, :, :t]
        x = x + self.org_model.freq_pos_embed[:, :, :, :]  # Just for sin pos embed
        x = torch.permute(torch.flatten(x, 2, 3), (0, 2, 1))  # rearrange(x, "b c f t -> b (f t) c")
        if self.org_model.pooling == "token":
            cls_token = self.org_model.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.org_model.token_pos_embed[:, :]
            x = torch.cat((cls_token, x), dim=1)
        x = self.org_model.pos_drop(x)

        return x, n_splits, padding_start

    def postprocess(self, x: torch.Tensor, n_splits: int, padding_start: int) -> torch.Tensor:
        x = self.org_model.norm(x)
        x = torch.reshape(x, (x.shape[0] // n_splits, -1, x.shape[-1]))
        if padding_start:
            x = x[:, :padding_start, :]
        return x

    def forward(self, wave_x: torch.Tensor, wave_n_ref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wave_x: [B, T]
            wave_n_ref: [B, T]
        Returns:
            x: [B, L, D]
        """
        self.validate_wave_pair(wave_x=wave_x, wave_n_ref=wave_n_ref)
        # Preprocess inputs [B, T] -> [B, 1, F=64, T]
        x = self.feature_extractor(wave_x, sampling_rate=self.sr, return_tensors="pt")
        x = x["input_values"].unsqueeze(1)
        n_ref = self.feature_extractor(wave_n_ref, sampling_rate=self.sr, return_tensors="pt")
        n_ref = n_ref["input_values"].unsqueeze(1)

        # extract patch embeddings [B, 1, F, T] -> [B, L, D]
        x, n_splits, padding_start = self.preprocess(x)
        n_ref, _, _ = self.preprocess(n_ref)

        # transformer blocks with fusion
        for i, blk in enumerate(self.org_model.blocks):
            x = blk(x)  # [B, L, D]
            n_ref = blk(n_ref)  # [B, L, D]

            # [B, L, D] -> [L, B, D] -> [B, L, D]
            x = self.fusion_layers[i](x=x.transpose(0, 1), n_ref=n_ref.transpose(0, 1)).transpose(0, 1)

        x = self.postprocess(x, n_splits, padding_start)
        return x
