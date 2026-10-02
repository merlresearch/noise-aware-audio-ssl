# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


def instantiate(
    cfg: Any,
    *args: Any,
    recursive: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Project-wide default Hydra instantiation.

    By default, Hydra's instantiate() is recursive. This wrapper makes
    non-recursive instantiation the default to reduce accidental nested object
    construction.

    Args:
        cfg: A config node (DictConfig / dataclass / plain dict) with _target_.
        *args: Positional args forwarded to Hydra instantiate.
        recursive: Whether to recursively instantiate nested configs.
        **kwargs: Keyword args forwarded to Hydra instantiate.

    Returns:
        Instantiated object.
    """
    return hydra_instantiate(cfg, *args, _recursive_=recursive, **kwargs)


def hydra_to_pydantic(hydra_cfg: DictConfig, cls: BaseModel) -> BaseModel:
    """Converts Hydra config to Pydantic config."""
    config_dict: dict[str, Any] = OmegaConf.to_object(hydra_cfg)  # type: ignore
    return cls(**config_dict)
