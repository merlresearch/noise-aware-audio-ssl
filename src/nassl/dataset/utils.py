# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import fcntl
import hashlib
import logging
import subprocess
from glob import glob
from pathlib import Path
from typing import Any

import torch
import torchaudio
from tqdm import tqdm

from nassl.utils.io import read_json

logger = logging.getLogger(__name__)


def parse_path_selector(selector: str) -> list[Path]:
    assert isinstance(selector, str)
    if selector.endswith(".json"):  # json format
        path_list = [Path(path) for path in read_json(json_path=selector)]
        logger.info(f"Loaded {len(path_list)} paths from {selector}")
    elif "*" in selector:  # glob format
        path_list = sorted(Path(path) for path in glob(selector))
        logger.info(f"Loaded {len(path_list)} paths from {selector}")
    elif selector.endswith(".wav"):  # single file format
        path_list = [Path(selector)]
    else:
        raise ValueError(f"Unknown path_list format: {selector}")
    return path_list


def _build_local_data_path(original_path: Path, local_cache_root: Path) -> Path:
    return local_cache_root / "files" / original_path.resolve().relative_to("/")


def _is_under_local_data_cache(path: Path, local_cache_root: Path) -> bool:
    cache_data_root = local_cache_root / "files"
    return path == cache_data_root or cache_data_root in path.parents


def _hash_path_list(path_list: list[Path]) -> str:
    sorted_path_list = sorted({path for path in path_list})
    content = "\n".join(str(path) for path in sorted_path_list)
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def _write_rsync_file_list(file_list_path: Path, resolved_path_list: list[Path]) -> None:
    with file_list_path.open("w") as file_obj:
        for resolved_path in tqdm(resolved_path_list):
            assert resolved_path.is_absolute()
            path_in_root = resolved_path.relative_to("/")
            file_obj.write(f"{path_in_root}\n")


def _stage_paths_with_rsync(resolved_path_list: list[Path], local_cache_root: Path) -> None:
    if len(resolved_path_list) == 0:
        return

    lock_dir = local_cache_root / ".stage_lock"
    lock_dir.mkdir(parents=True, exist_ok=True)

    list_hash = _hash_path_list(path_list=resolved_path_list)
    done_path = lock_dir / f"{list_hash}.done"
    lock_path = lock_dir / f"{list_hash}.lock"
    file_list_path = lock_dir / f"{list_hash}.txt"

    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not done_path.exists():
            _write_rsync_file_list(file_list_path=file_list_path, resolved_path_list=resolved_path_list)
            destination_root = local_cache_root / "files"
            destination_root.mkdir(parents=True, exist_ok=True)
            rsync_cmd = [
                "rsync",
                "-a",
                "--ignore-existing",
                "--files-from",
                str(file_list_path),
                "/",
                str(destination_root),
            ]
            logger.info(
                f"Staging {len(resolved_path_list)} files into local-data cache: {local_cache_root}"
                + f"\nExample: {resolved_path_list[0]}"
            )
            subprocess.run(rsync_cmd, check=True)
            done_path.touch()
        else:
            logger.info(
                f"Cache already staged for {len(resolved_path_list)} files in local-data cache: {local_cache_root}."
                + f"\nExample: {resolved_path_list[0]}"
            )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _resolve_path_list(path_list: list[Path], local_cache_root: Path) -> list[Path]:
    resolved_list = []
    for path in path_list:
        resolved_path = path.resolve()
        assert not _is_under_local_data_cache(path=resolved_path, local_cache_root=local_cache_root)
        resolved_list.append(resolved_path)
    return sorted(set(resolved_list))


def stage_paths_to_local_data(path_list: list[Path], local_cache_root: str | Path | None) -> list[Path]:
    if local_cache_root is None:
        return path_list
    else:
        local_cache_root = Path(local_cache_root).expanduser().resolve()

    resolved_path_list = _resolve_path_list(path_list=path_list, local_cache_root=local_cache_root)
    _stage_paths_with_rsync(resolved_path_list=resolved_path_list, local_cache_root=local_cache_root)

    return [_build_local_data_path(original_path=path, local_cache_root=local_cache_root) for path in path_list]


def _stage_manifest_item_paths(item_list: list[dict[str, Any]], local_cache_root: str | None) -> list[dict[str, Any]]:
    if local_cache_root is None:
        return item_list

    path_list: list[Path] = []
    for item in tqdm(item_list):
        for key, value in item.items():
            if key.endswith("_path"):
                path_list.append(Path(value))

    staged_path_list = stage_paths_to_local_data(path_list=path_list, local_cache_root=local_cache_root)
    path_map: dict[str, str] = {}
    for source_path, staged_path in zip(path_list, staged_path_list):
        path_map[str(source_path)] = str(staged_path)

    remapped_item_list: list[dict[str, Any]] = []
    for item in tqdm(item_list):
        remapped_item = dict(item)
        for key, value in item.items():
            if key.endswith("_path"):
                remapped_item[key] = path_map[str(Path(value))]
        remapped_item_list.append(remapped_item)
    return remapped_item_list


def torch_mono_wav_load(path: str | Path, sample_rate: int) -> torch.Tensor:
    wave, sr = torchaudio.load(path)
    assert sr == sample_rate and wave.shape[0] == 1
    return wave[0]


def build_feature_cache_path(
    wave_path: Path,
    feature_dir: Path,
    feature_ext: str,
    segment: tuple[int, int] | tuple[str, str] | None = None,
) -> Path:
    hash_base = str(wave_path.resolve())
    if segment is not None:
        hash_base += f"_{segment[0]}_{segment[1]}"
    path_hash = hashlib.sha1(hash_base.encode("utf-8")).hexdigest()
    return feature_dir / f"{wave_path.stem}_{path_hash}{feature_ext}"


def collect_paths(path_selector_list: list[str]) -> list[Path]:
    path_list: list[Path] = []
    logger.info("Start Loading Paths")
    for selector in path_selector_list:
        path_list.extend(parse_path_selector(selector=selector))
    logger.info("Finished Loading Paths")
    return path_list


def _avoid_zero_signal(
    wave_s: torch.Tensor,
    eps: float = 1e-7,
    source_path: str | Path | None = None,
) -> torch.Tensor:
    if wave_s.abs().sum() == 0:
        wave_s += eps
        message = "Loaded wave_s is all zeros, added small noise to avoid silent input"
        if source_path is not None:
            message += f" (source_path: {source_path})"
        logger.warning(message)
    return wave_s
