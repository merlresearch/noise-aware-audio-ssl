# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import torch

from nassl.model.beats.BEATs import BEATs, BEATsConfig
from nassl.utils.download import get_cached_model_path

from .base import BaseExtractor


class FE_BEATs(BaseExtractor):

    def __init__(self, model_id: str = "BEATs_iter3") -> None:
        super().__init__()

        ckpt_path = get_cached_model_path(model_id=model_id)
        checkpoint = torch.load(ckpt_path)
        logging.info(f"model_id: {model_id}")

        cfg = BEATsConfig(checkpoint["cfg"])
        self.backbone = BEATs(cfg)
        self.backbone.load_state_dict(checkpoint["model"])
        self.backbone.eval()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        padding_mask = torch.zeros_like(wave).bool()
        feature = self.backbone.extract_features(wave, padding_mask=padding_mask)[0]
        return feature  # B, L, D
