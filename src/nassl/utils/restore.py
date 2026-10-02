# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from nassl.utils.download import BASE_MODEL_LIST, get_cached_model_path
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


def _get_model_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    model_state_dict = {key[len("model.") :]: value for key, value in state_dict.items() if key.startswith("model.")}
    if not model_state_dict:
        raise ValueError("No keys starting with 'model.' were found in the checkpoint")
    return model_state_dict


def _get_trainable_prefix(model: torch.nn.Module) -> str:
    model_name = model.__class__.__name__
    if model_name == "NABEATs":
        trainable_prefix = "encoder.fusion_layers."
    elif model_name == "DBEATs":
        trainable_prefix = "encoder.additional_layers."
    elif model_name == "NADasheng":
        trainable_prefix = "fusion_layers."
    elif model_name == "NAEAT":
        trainable_prefix = "fusion_layers."
    else:
        raise NotImplementedError(f"Model name: {model_name}")
    return trainable_prefix


def _check_incompat(model: torch.nn.Module, incompat: torch.nn.modules.module._IncompatibleKeys) -> None:
    param_dict = dict(model.named_parameters())
    relative_attention_bias_pattern = re.compile(r"encoder\.layers\..*\.self_attn\.relative_attention_bias\.weight")

    # Check that all missing keys are either relative attention bias weights or frozen parameters
    for key in incompat.missing_keys:
        if relative_attention_bias_pattern.fullmatch(key):
            continue
        assert not param_dict[key].requires_grad

    # Check that all trainable parameters are known and not missing
    trainable_prefix = _get_trainable_prefix(model=model)
    for key, param in param_dict.items():
        if param.requires_grad:
            assert key.startswith(trainable_prefix)
            assert key not in incompat.missing_keys

    assert not incompat.unexpected_keys


def restore_model(
    ckpt_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    model_id: Optional[str] = None,
) -> torch.nn.Module:
    """Restore a model from a local checkpoint or a published model.

    Args:
        ckpt_path: Path to a local checkpoint file.
        device: Device on which to restore the model.
        model_id: Identifier of a published pretrained model. The checkpoint is
            downloaded to the NASSL cache when it is not already cached.

    Returns:
        The restored model in evaluation mode.

    Raises:
        ValueError: If both or neither of ``ckpt_path`` and ``model_id`` are
            specified, or if ``model_id`` is unknown.
    """
    if (ckpt_path is None) == (model_id is None):
        raise ValueError("Specify exactly one of ckpt_path or model_id")
    if model_id is not None:
        if model_id in BASE_MODEL_LIST:
            raise ValueError(f"Model '{model_id}' is a base SSL model and cannot be restored with this function")
        ckpt_path = get_cached_model_path(model_id=model_id)
    assert ckpt_path is not None
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model = instantiate(cfg=ckpt["hyper_parameters"]["model_cfg"])
    model_state_dict = _get_model_state_dict(state_dict=ckpt["state_dict"])
    incompat = model.load_state_dict(model_state_dict, strict=False)
    _check_incompat(model=model, incompat=incompat)
    model.to(device)
    model.eval()
    return model
