# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nassl.dataset import TensorCollator, TestDataset
from nassl.dataset.utils import build_feature_cache_path
from nassl.utils.hydra import hydra_to_pydantic, instantiate

logger = logging.getLogger(__name__)
CONFIG_PATH = "../../../../config/preprocessing/feature_extraction"


class Config(BaseModel):
    seed: int
    save_dir: Path
    feature_ext: str = ".pt"
    sample_rate: int = 16000
    batch_size: int = 1
    device: str = "cpu"
    feature_extractor_cfg: dict[str, Any]
    manifest_test_path: Path
    wave_key: str


def preprocess_feature_cache(
    manifest_json_path: Path,
    save_dir: Path,
    feature_extractor: nn.Module,
    sample_rate: int,
    batch_size: int,
    device: str,
    feature_ext: str,
    wave_key: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TestDataset(
        manifest_json_path=str(manifest_json_path),
        sample_rate=sample_rate,
        return_wave_s=True,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=TensorCollator(),
    )

    for batch in tqdm(dataloader):
        wave_batch: torch.Tensor = batch[f"wave_{wave_key}"].to(device)
        if wave_key != "x":
            segment_batch = batch[f"segment_{wave_key}"]
        else:
            raise NotImplementedError
        with torch.no_grad():
            feature_batch = feature_extractor(wave_batch).detach().cpu()
        for wave_path, feature, seg in zip(batch[f"wave_{wave_key}_path"], feature_batch, segment_batch):
            feature_path = build_feature_cache_path(
                wave_path=Path(wave_path).resolve(),
                feature_dir=save_dir,
                feature_ext=feature_ext,
                segment=seg,
            )
            feature_to_save: torch.Tensor = feature.clone()
            # Clone each sample before saving so per-sample cache files do not share batch storage.
            torch.save(feature_to_save, feature_path)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: Config = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=Config)
    if cfg.save_dir.exists():
        raise FileExistsError(f"{cfg.save_dir} already exists. Please remove it before running this script.")
    seed_everything(cfg.seed)

    feature_extractor = instantiate(cfg=cfg.feature_extractor_cfg)
    feature_extractor = feature_extractor.to(cfg.device)
    feature_extractor.eval()
    preprocess_feature_cache(
        manifest_json_path=cfg.manifest_test_path,
        save_dir=cfg.save_dir,
        feature_extractor=feature_extractor,
        sample_rate=cfg.sample_rate,
        batch_size=cfg.batch_size,
        device=cfg.device,
        feature_ext=cfg.feature_ext,
        wave_key=cfg.wave_key,
    )


if __name__ == "__main__":
    main()
