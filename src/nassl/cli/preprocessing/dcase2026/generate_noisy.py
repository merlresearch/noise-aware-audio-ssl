# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torchaudio
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from nassl.utils.hydra import hydra_to_pydantic, instantiate
from nassl.utils.io import write_json

CONFIG_PATH = "../../../../../config/" + "/".join(Path(__file__).with_suffix("").parts[-3:])

TEST_ITEM_KEYS = (
    "wave_x_path",
    "wave_n_ref_path",
    "wave_s_path",
    "segment_x",
    "segment_n_ref",
    "segment_s",
)


class Config(BaseModel):
    split: str
    clean_parent_dir_list: list[Path]
    dataset: dict[str, Any]
    collator: dict[str, Any]
    output_dir: Path
    name: str
    fs: int = 16000
    seed: int = 0
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False


def get_output_split_dir(cfg: Config) -> Path:
    split_dir = cfg.output_dir / cfg.name / cfg.split
    if split_dir.exists():
        raise FileExistsError(f"Output directory already exists: {split_dir}")
    return split_dir


def get_channel_output_paths(
    split_dir: Path, wave_s_path: Path, clean_parent_dir_list: list[Path]
) -> tuple[Path, Path, Path]:
    relative_wave_s_path = get_relative_wave_s_path(
        wave_s_path=wave_s_path, clean_parent_dir_list=clean_parent_dir_list
    )
    sample_dir = split_dir / "audio" / relative_wave_s_path
    sample_dir = sample_dir.parent / sample_dir.stem
    sample_dir.mkdir(parents=True, exist_ok=False)
    return (
        sample_dir / "noisy.wav",
        sample_dir / "noise_ref.wav",
        sample_dir / "clean.wav",
    )


def get_relative_wave_s_path(wave_s_path: Path, clean_parent_dir_list: list[Path]) -> Path:
    for clean_parent_dir in clean_parent_dir_list:
        try:
            return wave_s_path.resolve().relative_to(clean_parent_dir.resolve())
        except ValueError:
            continue
    raise ValueError(f"Failed to resolve relative path for {wave_s_path}")


def build_dataloader(cfg: Config) -> DataLoader[Any]:
    dataset = instantiate(cfg=cfg.dataset)
    collator = instantiate(cfg=cfg.collator)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collator,
    )


def append_batch_to_manifests(
    batch: dict[str, Any],
    split_dir: Path,
    clean_parent_dir_list: list[Path],
    fs: int,
    manifests: dict[str, list[dict[str, Any]]],
) -> None:
    batch_size = int(batch["wave_x"].shape[0])
    for idx in range(batch_size):
        wave_s_path = Path(batch["wave_s_path"][idx])
        wave_x_path, wave_n_ref_path, wave_s_output_path = get_channel_output_paths(
            split_dir=split_dir,
            wave_s_path=wave_s_path,
            clean_parent_dir_list=clean_parent_dir_list,
        )
        wave_x = batch["wave_x"][idx : idx + 1].detach().cpu()
        wave_n_ref = batch["wave_n_ref"][idx : idx + 1].detach().cpu()
        wave_s = batch["wave_s"][idx : idx + 1].detach().cpu()
        torchaudio.save(wave_x_path, wave_x, fs)
        torchaudio.save(wave_n_ref_path, wave_n_ref, fs)
        torchaudio.save(wave_s_output_path, wave_s, fs)
        segment_s = (0, int(wave_s.shape[-1]))
        segment_n_ref = (0, int(wave_n_ref.shape[-1]))
        segment_x = (0, int(wave_x.shape[-1]))
        pair_manifest = batch["pair_manifest"][idx]

        noisy_manifest_item: dict[str, Any] = {
            "wave_x_path": str(wave_x_path),
            "wave_n_ref_path": str(wave_n_ref_path),
            "wave_s_path": str(wave_s_output_path),
            "original_wave_s_path": batch["wave_s_path"][idx],
            "original_segment_s": batch["segment_s"][idx],
            "wave_n_path": batch["wave_n_path"][idx],
            "segment_x": segment_x,
            "segment_n_ref": segment_n_ref,
            "segment_s": segment_s,
            "segment_n": batch["segment_n"][idx],
            "snr_db": pair_manifest["snr_db"],
            "rir_sample_dir": pair_manifest["rir_sample_dir"],
            "rir_metadata": pair_manifest["rir_metadata"],
        }
        manifests["noisy"].append(noisy_manifest_item)
        manifests["test"].append({k: noisy_manifest_item[k] for k in TEST_ITEM_KEYS})


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: Config = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=Config)
    seed_everything(cfg.seed)
    split_dir = get_output_split_dir(cfg=cfg)
    dataloader = build_dataloader(cfg=cfg)
    manifests: dict[str, list[dict[str, Any]]] = {"test": [], "noisy": []}

    for batch in tqdm(dataloader):
        append_batch_to_manifests(
            batch=batch,
            split_dir=split_dir,
            clean_parent_dir_list=cfg.clean_parent_dir_list,
            fs=cfg.fs,
            manifests=manifests,
        )

    for key, manifest in manifests.items():
        write_json(json_path=split_dir / f"manifest_{key}.json", data=manifest)


if __name__ == "__main__":
    main()
