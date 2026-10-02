# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import torch

from nassl.model.eat.tools import preprocess, restore

from .base import BaseExtractor


class FE_EAT(BaseExtractor):

    def __init__(self, model_id: str = "worstchan/EAT-base_epoch30_pretrain") -> None:
        super().__init__()

        logging.info(f"model_id: {model_id}")

        self.backbone, self.feature_dim = restore(model_id=model_id)
        self.backbone.eval()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        x = preprocess(wave)
        feature = self.backbone.extract_features(x)
        return feature  # B, L, D
