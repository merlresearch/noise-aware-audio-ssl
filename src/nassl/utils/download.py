# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import os
from pathlib import Path

from filelock import FileLock
from torch.hub import download_url_to_file

logger = logging.getLogger(__name__)

GITHUB_RELEASE_URL = "https://github.com/merlresearch/noise-aware-audio-ssl/releases/download"

BASE_MODEL_DICT: dict[str, list[str]] = {"v1.0.0": ["BEATs_iter3"]}
NA_MODEL_DICT: dict[str, list[str]] = {
    "v1.0.0": [
        "iwaenc2026_dbeats_seed0",
        "iwaenc2026_nabeats_ca_seed0",
        "iwaenc2026_nabeats_film_seed0",
        "dcase2026_nabeats_seed0",
        "dcase2026_nabeats_seed1",
        "dcase2026_nabeats_seed2",
        "dcase2026_naeat_seed0",
        "dcase2026_naeat_seed1",
        "dcase2026_naeat_seed2",
        "dcase2026_nadasheng_seed0",
        "dcase2026_nadasheng_seed1",
        "dcase2026_nadasheng_seed2",
    ],
}
BASE_MODEL_LIST = [model for models in BASE_MODEL_DICT.values() for model in models]
NA_MODEL_LIST = [model for models in NA_MODEL_DICT.values() for model in models]

URL_DICT: dict[str, str] = {}
for tag, models in BASE_MODEL_DICT.items():
    for model in models:
        if model == "BEATs_iter3":
            URL_DICT[model] = f"{GITHUB_RELEASE_URL}/{tag}/{model}.pt"
        else:
            raise NotImplementedError(f"Unknown base model: {model}")
for tag, models in NA_MODEL_DICT.items():
    for model in models:
        URL_DICT[model] = f"{GITHUB_RELEASE_URL}/{tag}/{model}.ckpt"


def get_cache_dir() -> Path:
    cache_dir = Path(os.environ.get("NASSL_CACHE_DIR", "~/.cache/nassl")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cached_model_path(model_id: str) -> Path:
    if model_id not in URL_DICT:
        raise ValueError(f"Unknown model_id: '{model_id}'")

    model_path = get_cache_dir() / Path(URL_DICT[model_id]).name
    if model_path.exists():
        return model_path
    lock_path = Path(f"{model_path}.lock")
    tmp_path = Path(f"{model_path}.tmp")

    with FileLock(str(lock_path)):
        if model_path.exists():
            return model_path
        logger.info(f"Downloading pretrained model '{model_id}' to '{model_path}'")
        try:
            download_url_to_file(url=URL_DICT[model_id], dst=str(tmp_path), progress=True)
            tmp_path.replace(model_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return model_path
