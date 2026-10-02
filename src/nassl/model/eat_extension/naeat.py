# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2024 Wenxi Chen
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Adapted from EAT: https://github.com/cwx-worst-one/EAT

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn

from nassl.model.eat.tools import preprocess, restore
from nassl.model.noiseaware import NoiseAwareModel
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


class NAEAT(NoiseAwareModel):
    def __init__(
        self,
        fusion_layer_cfg: dict[str, Any],
        model_id: str = "worstchan/EAT-base_epoch30_pretrain",
        trainable_base: bool = True,
        eval_base: bool = True,
    ) -> None:
        super().__init__()

        self.trainable_base = trainable_base
        self.eval_base = eval_base

        # Load the original EAT model
        _wrapper, self.embedding_dim = restore(model_id=model_id)
        self.org_model = _wrapper.model
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

    def train(self, mode: bool = True) -> NAEAT:
        super().train(mode=mode)
        self._set_frozen_base_eval()
        return self

    def extract_patch_embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: mel spectrogram [B, 1, T, F]
        Returns:
            x: patch embeddings [B, L, D]
        """
        B = x.shape[0]
        x = self.org_model.local_encoder(x)  # flattened to [B, L, D]
        if self.org_model.fixed_positional_encoder is not None:
            x = x + self.org_model.fixed_positional_encoder(x, None)[:, : x.size(1), :]
        x = torch.cat((self.org_model.extra_tokens.expand(B, -1, -1), x), dim=1)
        x = self.org_model.pre_norm(x)
        x = self.org_model.pos_drop(x)
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
        # Preprocess inputs [B, T] -> [B, 1, T, F]
        x = preprocess(wave_x)
        n_ref = preprocess(wave_n_ref)

        # extract patch embeddings [B, 1, T, F] -> [B, L, D]
        x = self.extract_patch_embed(x)
        n_ref = self.extract_patch_embed(n_ref)

        # transformer blocks with fusion
        for i, blk in enumerate(self.org_model.blocks):
            x, _ = blk(x)  # [B, L, D]
            n_ref, _ = blk(n_ref)  # [B, L, D]

            # [B, L, D] -> [L, B, D] -> [B, L, D]
            x = self.fusion_layers[i](x=x.transpose(0, 1), n_ref=n_ref.transpose(0, 1)).transpose(0, 1)

        return x
