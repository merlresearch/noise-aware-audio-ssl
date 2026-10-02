# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def load_beats_pretrained_state(
    model: nn.Module,
    checkpoint: dict[str, Any],
    additional_layer_prefix: str,
    trainable_base: bool,
) -> None:
    load_result = model.load_state_dict(checkpoint["model"], strict=False)
    if load_result.unexpected_keys:
        raise ValueError(f"Unexpected BEATs keys: {load_result.unexpected_keys}")

    missing_keys = [key for key in load_result.missing_keys if not key.startswith(additional_layer_prefix)]
    if missing_keys:
        raise ValueError(f"Missing BEATs keys: {missing_keys}")

    for name, param in model.named_parameters():
        if name in checkpoint["model"] and torch.all(param == checkpoint["model"][name]):
            param.requires_grad = trainable_base
        elif not name.startswith(additional_layer_prefix):
            raise ValueError(
                f"Parameter '{name}' is neither loaded from the BEATs checkpoint "
                f"nor part of the additional layers with prefix '{additional_layer_prefix}'."
            )
