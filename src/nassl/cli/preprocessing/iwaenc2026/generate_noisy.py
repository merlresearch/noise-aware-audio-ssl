# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import hydra
import torch
import torchaudio
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm import tqdm

from nassl.cli.preprocessing.iwaenc2026.modify_manifest_test import ShiftSegmentNRefModificator, modify_manifest_test
from nassl.dataset import NoisePairShiftLeftFromEndDataset
from nassl.module.signal_processing import mix_batch_snr
from nassl.utils.hydra import hydra_to_pydantic
from nassl.utils.io import write_json

logger = logging.getLogger(__name__)
CONFIG_PATH = "../../../../../config/" + "/".join(Path(__file__).with_suffix("").parts[-3:])


class TargetConfig(BaseModel):
    path_selector_list: list[str]
    original_parent_dir: Path


class NoiseConfig(BaseModel):
    path_selector_list: list[str]


class Config(BaseModel):
    split: str
    target: TargetConfig
    noise: NoiseConfig
    snr_range: tuple[float, float]
    output_dir: Path
    sec_s: float
    sec_n_ref: float | str = "same_as_s"
    seed: int = 0
    sample_rate: int = 16000
    create_shift_left_slen: bool = True


def get_wave_x_path(save_dir: Path, wave_s_path: Path, original_parent_dir: Path) -> Path:
    wave_s_relative_path = wave_s_path.resolve().relative_to(original_parent_dir)
    wave_x_path = save_dir / wave_s_relative_path
    wave_x_path.parent.mkdir(parents=True, exist_ok=True)

    if wave_x_path.exists():
        raise FileExistsError(f"File already exists: {wave_x_path}")
    return wave_x_path


def mix_wave_s_wave_n_with_random_snr(
    wave_s: torch.Tensor,
    wave_n: torch.Tensor,
    snr_range: tuple[float, float],
    rng: random.Random,
) -> tuple[torch.Tensor, float]:
    assert wave_s.ndim == 1 and wave_n.ndim == 1
    assert wave_s.shape == wave_n.shape
    snr = rng.uniform(snr_range[0], snr_range[1])
    snr_tensor = torch.tensor([snr], device=wave_s.device, dtype=wave_s.dtype)
    wave_x = mix_batch_snr(wave_s=wave_s.unsqueeze(0), wave_n=wave_n.unsqueeze(0), snr=snr_tensor).squeeze(0)
    return wave_x, snr


def generate_test_manifest_with_noisy_dataset(
    pair_dataset: Dataset,
    save_dir: Path,
    sample_rate: int,
    snr_range: tuple[float, float],
    original_parent_dir: Path,
    seed: int,
) -> None:
    rng = random.Random(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    item_list_test: list[dict[str, Any]] = []
    item_list_noisy: list[dict[str, Any]] = []
    logger.info(f"Generating noisy test data and manifest for {len(pair_dataset)} samples...")
    for idx in tqdm(range(len(pair_dataset))):
        sample = pair_dataset[idx]
        wave_x, snr_db = mix_wave_s_wave_n_with_random_snr(
            wave_s=sample["wave_s"],
            wave_n=sample["wave_n"],
            snr_range=snr_range,
            rng=rng,
        )

        wave_x_path = get_wave_x_path(
            save_dir=save_dir / "audio",
            wave_s_path=Path(sample["wave_s_path"]),
            original_parent_dir=original_parent_dir,
        )
        torchaudio.save(wave_x_path, wave_x.unsqueeze(0), sample_rate=sample_rate)

        item_test: dict[str, Any] = {"wave_x_path": str(Path(wave_x_path).resolve())}
        for key in ["wave_n_ref_path", "wave_s_path"]:
            item_test[key] = str(Path(sample[key]).resolve())
        for key in ["segment_n_ref", "segment_s"]:
            item_test[key] = sample[key]
        item_list_test.append(item_test)

        item_noisy: dict[str, Any] = {"wave_x_path": str(Path(wave_x_path).resolve())}
        for key in ["wave_n_path", "wave_s_path"]:
            item_noisy[key] = str(Path(sample[key]).resolve())
        for key in ["segment_n", "segment_s"]:
            item_noisy[key] = sample[key]
        item_noisy["snr_db"] = snr_db
        item_list_noisy.append(item_noisy)

    manifest_test_path = save_dir / "manifest_test.json"
    manifest_noisy_path = save_dir / "manifest_noisy.json"
    write_json(json_path=manifest_test_path, data=item_list_test)
    write_json(json_path=manifest_noisy_path, data=item_list_noisy)


def generate_noisy(
    save_dir: Path,
    target_path_selector_list: list[str],
    noise_path_selector_dict: dict[str, list[str]],
    snr_range: tuple[float, float],
    original_parent_dir: Path,
    sec_s: float,
    min_start_sec_n: float,
    sec_n_ref: float | str = "same_as_s",
    seed: int = 0,
    sample_rate: int = 16000,
) -> None:
    seed_everything(seed)
    if (save_dir / "audio").exists():
        raise FileExistsError(f"{save_dir} already exists. Please remove it before running this script.")
    validator_cfg = {"_target_": "nassl.dataset.validator.NoisePowerValidator"}
    pair_dataset = NoisePairShiftLeftFromEndDataset(
        target_path_selector_list=target_path_selector_list,
        noise_path_selector_dict=noise_path_selector_dict,
        sec_s=sec_s,
        validator_cfg=validator_cfg,
        sample_rate=sample_rate,
        return_path=True,
        shuffle=False,
        max_retries=100,
        local_cache_root=None,
        min_start_sec_n=min_start_sec_n,
        sec_n_ref=sec_n_ref,
        sec_n_ref_shift=0,
    )
    generate_test_manifest_with_noisy_dataset(
        pair_dataset=pair_dataset,
        save_dir=save_dir,
        sample_rate=sample_rate,
        snr_range=snr_range,
        original_parent_dir=original_parent_dir.resolve(),
        seed=seed,
    )


def generate_shift_left_slen_manifest(save_dir: Path, sample_rate: int, sec_s: float) -> None:
    modificator = ShiftSegmentNRefModificator(shift_sec=sec_s, shift_direction="left")
    modify_manifest_test(
        manifest_test_path=save_dir / "manifest_test.json",
        manifest_output_path=save_dir / "manifest_test_shift_left_slen.json",
        sample_rate=sample_rate,
        modificator=modificator,
    )


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: Config = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=Config)
    generate_noisy(
        save_dir=cfg.output_dir,
        target_path_selector_list=cfg.target.path_selector_list,
        noise_path_selector_dict={"noise": cfg.noise.path_selector_list},
        snr_range=cfg.snr_range,
        original_parent_dir=cfg.target.original_parent_dir,
        sec_s=cfg.sec_s,
        min_start_sec_n=cfg.sec_s,
        sec_n_ref=cfg.sec_n_ref,
        seed=cfg.seed,
        sample_rate=cfg.sample_rate,
    )
    if cfg.create_shift_left_slen:
        generate_shift_left_slen_manifest(save_dir=cfg.output_dir, sample_rate=cfg.sample_rate, sec_s=cfg.sec_s)


if __name__ == "__main__":
    main()
