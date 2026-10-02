# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import List

import pytest
import torch

from nassl.model.noiseaware import NoiseAwareModel
from nassl.utils.download import get_cache_dir
from nassl.utils.restore import restore_model

CHECKPOINT_CASES: List[str] = [
    "iwaenc2026_dbeats_seed0",
    "iwaenc2026_nabeats_ca_seed0",
    "iwaenc2026_nabeats_film_seed0",
    "dcase2026_nabeats_seed0",
    "dcase2026_naeat_seed0",
    "dcase2026_nadasheng_seed0",
]


def test_restore_model_requires_exactly_one_source() -> None:
    with pytest.raises(ValueError, match="Specify exactly one"):
        restore_model()

    with pytest.raises(ValueError, match="Specify exactly one"):
        restore_model(ckpt_path="model.ckpt", model_id="iwaenc2026_dbeats_seed0")


def _require_checkpoint(model_id: str) -> None:
    ckpt_path = get_cache_dir() / f"{model_id}.ckpt"
    if not ckpt_path.is_file() or ckpt_path.stat().st_size < 1024:
        pytest.skip(f"Checkpoint is unavailable: {ckpt_path}")


@pytest.mark.checkpoint
@pytest.mark.parametrize("model_id", CHECKPOINT_CASES)
def test_pretrained_checkpoint_forward(model_id: str) -> None:
    torch.manual_seed(0)
    _require_checkpoint(model_id=model_id)
    model = restore_model(model_id=model_id, device="cpu")
    wave_x = torch.randn(1, 16000)

    with torch.inference_mode():
        if isinstance(model, NoiseAwareModel):
            output = model(wave_x=wave_x, wave_n_ref=wave_x)
        else:
            output = model(wave_x)

    assert output.ndim == 3
    assert output.shape[0] == wave_x.shape[0]
