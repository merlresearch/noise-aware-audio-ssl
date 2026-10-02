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
from nassl.utils.download import get_cached_model_path
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


class TransformerEncoderAdditionalLayer(TransformerEncoder):
    def __init__(self, args: Any, additional_layer_cfg: dict[str, Any]) -> None:
        super().__init__(args=args)

        additional_layer_cfg = {**additional_layer_cfg, "embed_dim": self.embedding_dim}
        self.additional_layers = nn.ModuleList([instantiate(additional_layer_cfg) for _ in range(args.encoder_layers)])

    def extract_features(self, x, padding_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio != 1.0:
                x = GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    pos_bias=pos_bias,
                )
                x = self.additional_layers[i](x=x, padding_mask=padding_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class DBEATs(BEATs):
    def __init__(
        self,
        additional_layer_cfg: dict[str, Any],
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
        self.encoder = TransformerEncoderAdditionalLayer(args=cfg, additional_layer_cfg=additional_layer_cfg)

        load_beats_pretrained_state(
            model=self,
            checkpoint=checkpoint,
            additional_layer_prefix="encoder.additional_layers.",
            trainable_base=trainable_base,
        )
        self._set_frozen_base_eval(mode=True)

    def _set_frozen_base_eval(self, mode: bool) -> None:
        if self.trainable_base or not self.eval_base:
            return
        self.dropout_input.eval()
        self.encoder.eval()
        self.encoder.additional_layers.train(mode=mode)
        assert self.predictor is None

    def train(self, mode: bool = True) -> DBEATs:
        super().train(mode=mode)
        self._set_frozen_base_eval(mode=mode)
        return self

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        feature = self.extract_features(wave)[0]
        return feature  # B, L, D
