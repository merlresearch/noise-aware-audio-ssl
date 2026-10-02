# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2022 Microsoft
# Copyright (c) 2019 Facebook, Inc. and its affiliates.
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Adapted from BEATs: https://github.com/microsoft/unilm/tree/master/beats

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nassl.model.beats.backbone import GradMultiply, TransformerEncoder
from nassl.model.beats.BEATs import BEATs, BEATsConfig
from nassl.model.beats_extension.tools import load_beats_pretrained_state
from nassl.model.noiseaware import NoiseAwareModel
from nassl.utils.download import get_cached_model_path
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


class TransformerEncoderFusionLayer(TransformerEncoder):
    def __init__(self, args: Any, fusion_layer_cfg: dict[str, Any]) -> None:
        super().__init__(args=args)
        fusion_layer_cfg = {**fusion_layer_cfg, "embed_dim": self.embedding_dim}
        self.fusion_layers = nn.ModuleList([instantiate(fusion_layer_cfg) for _ in range(args.encoder_layers)])

    def _prepare_encoder_input(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if padding_mask is not None:
            x[padding_mask] = 0

        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return x.transpose(0, 1)  # B x T x C -> T x B x C

    def forward_with_ref(
        self,
        x: torch.Tensor,
        n_ref: torch.Tensor,
        padding_mask_x: torch.Tensor | None = None,
        padding_mask_n_ref: torch.Tensor | None = None,
        layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor | None]]]:
        x, n_ref, layer_results = self.extract_features_with_ref(
            x=x,
            n_ref=n_ref,
            padding_mask_x=padding_mask_x,
            padding_mask_n_ref=padding_mask_n_ref,
            tgt_layer=layer,
        )

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
            n_ref = self.layer_norm(n_ref)

        return x, n_ref, layer_results

    def extract_features_with_ref(
        self,
        x: torch.Tensor,
        n_ref: torch.Tensor,
        padding_mask_x: torch.Tensor | None = None,
        padding_mask_n_ref: torch.Tensor | None = None,
        tgt_layer: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor | None]]]:

        x = self._prepare_encoder_input(x=x, padding_mask=padding_mask_x)
        n_ref = self._prepare_encoder_input(x=n_ref, padding_mask=padding_mask_n_ref)

        layer_results: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        z_x: torch.Tensor | None = None
        if tgt_layer is not None:
            layer_results.append((x, z_x))

        r_x: torch.Tensor | None = None
        r_n_ref: torch.Tensor | None = None
        pos_bias_x: torch.Tensor | None = None
        pos_bias_n_ref: torch.Tensor | None = None
        for layer_idx, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
                n_ref = GradMultiply.apply(n_ref, self.layer_wise_gradient_decay_ratio)

            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                # embed: T x B x C
                x, z_x, pos_bias_x = layer(
                    x,
                    self_attn_padding_mask=padding_mask_x,
                    need_weights=False,
                    pos_bias=pos_bias_x,
                )
                n_ref, _, pos_bias_n_ref = layer(
                    n_ref,
                    self_attn_padding_mask=padding_mask_n_ref,
                    need_weights=False,
                    pos_bias=pos_bias_n_ref,
                )
                x = self.fusion_layers[layer_idx](
                    x=x,
                    n_ref=n_ref,
                    padding_mask_n_ref=padding_mask_n_ref,
                    pos_bias=pos_bias_x,
                )
            if tgt_layer is not None:
                layer_results.append((x, z_x))
            if layer_idx == tgt_layer:
                r_x = x
                r_n_ref = n_ref
                break

        if r_x is not None and r_n_ref is not None:
            x = r_x
            n_ref = r_n_ref

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        n_ref = n_ref.transpose(0, 1)
        return x, n_ref, layer_results


class NABEATs(BEATs, NoiseAwareModel):
    def __init__(
        self,
        fusion_layer_cfg: dict[str, Any],
        model_id: str = "BEATs_iter3",
        trainable_base: bool = True,
        eval_base: bool = True,
    ) -> None:
        ckpt_path = get_cached_model_path(model_id=model_id)
        checkpoint: dict[str, Any] = torch.load(ckpt_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        super().__init__(cfg=cfg)
        self.trainable_base = trainable_base
        self.eval_base = eval_base
        self.encoder = TransformerEncoderFusionLayer(args=cfg, fusion_layer_cfg=fusion_layer_cfg)

        load_beats_pretrained_state(
            model=self,
            checkpoint=checkpoint,
            additional_layer_prefix="encoder.fusion_layers.",
            trainable_base=trainable_base,
        )
        self._set_frozen_base_eval(mode=True)

    def _set_frozen_base_eval(self, mode: bool) -> None:
        if self.trainable_base or not self.eval_base:
            return
        self.dropout_input.eval()
        self.encoder.eval()
        self.encoder.fusion_layers.train(mode=mode)
        assert self.predictor is None

    def train(self, mode: bool = True) -> NABEATs:
        super().train(mode=mode)
        self._set_frozen_base_eval(mode=mode)
        return self

    def _extract_patch_features(
        self,
        source: torch.Tensor,
        padding_mask: torch.Tensor | None,
        fbank_mean: float,
        fbank_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)
        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)  # [B, L, D]
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        return features, padding_mask

    def extract_features(
        self,
        wave_x: torch.Tensor,
        wave_n_ref: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        padding_mask_n_ref: torch.Tensor | None = None,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, padding_mask = self._extract_patch_features(
            source=wave_x,
            padding_mask=padding_mask,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
        )
        n_ref, padding_mask_n_ref = self._extract_patch_features(
            source=wave_n_ref,
            padding_mask=padding_mask_n_ref,
            fbank_mean=fbank_mean,
            fbank_std=fbank_std,
        )

        x, n_ref, _ = self.encoder.forward_with_ref(
            x=x,
            n_ref=n_ref,
            padding_mask_x=padding_mask,
            padding_mask_n_ref=padding_mask_n_ref,
        )

        if self.predictor is not None:
            x = self.predictor_dropout(x)
            logits = self.predictor(x)
            if padding_mask is not None and padding_mask.any():
                logits[padding_mask] = 0
                logits = logits.sum(dim=1)
                logits = logits / (~padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits = logits.mean(dim=1)
            lprobs = torch.sigmoid(logits)
            return lprobs, padding_mask
        else:
            return x, padding_mask

    def forward(self, wave_x: torch.Tensor, wave_n_ref: torch.Tensor) -> torch.Tensor:
        self.validate_wave_pair(wave_x=wave_x, wave_n_ref=wave_n_ref)
        x, _ = self.extract_features(wave_x=wave_x, wave_n_ref=wave_n_ref)
        return x
