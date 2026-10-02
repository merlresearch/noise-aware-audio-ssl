# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from transformers import AutoFeatureExtractor, AutoModel

from .base import BaseExtractor


class FE_Dasheng(BaseExtractor):
    def __init__(self, model_id: str = "mispeech/dasheng-base") -> None:
        super().__init__()
        self.sr = 16000

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_id, outputdim=None, trust_remote_code=True)
        self.backbone.eval()

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        inputs = self.feature_extractor(wave, sampling_rate=self.sr, return_tensors="pt")
        outputs = self.backbone(**inputs)
        feature = outputs.hidden_states
        return feature  # B, L, D
