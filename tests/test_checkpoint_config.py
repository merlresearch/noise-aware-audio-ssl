# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import io
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel

from nassl.utils.checkpoint import to_checkpoint_config


class Mode(Enum):
    TRAIN = "train"


class Config(BaseModel):
    path: Path


def test_checkpoint_config_weights_only_roundtrip() -> None:
    cfg = OmegaConf.create({"width": 8, "model": {"width": "${width}"}})
    tensor = torch.arange(3)
    converted = to_checkpoint_config(
        {
            "cfg": cfg,
            "nested": [Config(path=Path("model.ckpt")), (Mode.TRAIN, np.int64(3))],
            "array": np.array([1.0, 2.0]),
            "tensor": tensor,
            "optional": None,
        }
    )
    stream = io.BytesIO()
    torch.save({"hyper_parameters": converted}, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)["hyper_parameters"]

    assert loaded["cfg"] == {"width": 8, "model": {"width": 8}}
    assert loaded["nested"] == [{"path": "model.ckpt"}, ("train", 3)]
    assert loaded["array"] == [1.0, 2.0]
    assert loaded["optional"] is None
    torch.testing.assert_close(loaded["tensor"], tensor)
    assert converted["tensor"] is tensor
    assert OmegaConf.is_config(cfg)


def test_checkpoint_config_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError, match="Unsupported checkpoint configuration type: object"):
        to_checkpoint_config({"unsupported": object()})
