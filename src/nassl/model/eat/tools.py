# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2024 Wenxi Chen
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT
#
# Adapted from EAT: https://github.com/cwx-worst-one/EAT

import logging
from typing import Tuple

import torch
import torchaudio.compliance.kaldi as ta_kaldi
from transformers import AutoModel

logger = logging.getLogger(__name__)

FBANK_PRAMS = {
    "htk_compat": True,
    "sample_frequency": 16000,
    "use_energy": False,
    "window_type": "hanning",
    "num_mel_bins": 128,
    "dither": 0.0,
    "frame_shift": 10,
}


def preprocess(
    source: torch.Tensor,
    norm_mean: float = -4.268,
    norm_std: float = 4.569,
) -> torch.Tensor:
    """
    Args:
        source (torch.Tensor): Waveforms with shape (B, L).

    Returns:
        fbank (torch.Tensor): Filter-bank features with shape (B, 1, T, F).
            T is padded to a multiple of 32.
    """
    if source.ndim != 2:
        raise ValueError(f"source must have shape (B, L), but got {tuple(source.shape)}")

    source = source - source.mean(dim=-1, keepdim=True)  # (B, L)

    # Compute filter-bank features
    fbanks = [ta_kaldi.fbank(waveform.unsqueeze(0), **FBANK_PRAMS) for waveform in source]
    fbank = torch.stack(fbanks, dim=0).unsqueeze(1)  # (B, 1, T, F)

    # Pad T to a multiple of 32
    # EAT requires target_length to be a multiple of 16, and recommends
    # target_length=1024 for 10-second audio. A multiple of 32 satisfies both requirements.
    # Reference: https://github.com/cwx-worst-one/EAT/blob/main/feature_extract/readme.md
    current_length = fbank.shape[-2]
    target_length = ((current_length + 31) // 32) * 32
    diff = target_length - current_length
    if diff > 0:
        fbank = torch.nn.functional.pad(
            fbank,
            pad=(0, 0, 0, diff),
            mode="constant",
            value=0,
        )

    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank


def restore(model_id: str) -> Tuple[torch.nn.Module, int]:
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    feature_dim = model.model.config.embed_dim
    logger.info(f"Restored EAT model from {model_id}, feature_dim: {feature_dim}")
    return model, feature_dim
