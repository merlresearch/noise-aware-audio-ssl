# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from nassl.dataset.pl_datamodule import PLDataModule
from nassl.utils.callback import NaNCheckCallback
from nassl.utils.config_class.train import TrainConfig
from nassl.utils.hydra import hydra_to_pydantic, instantiate

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*You are using `torch\.load` with `weights_only=False`.*",
)
logger = logging.getLogger(__name__)


def make_trainer(cfg: TrainConfig, ckpt_dir: Path) -> pl.Trainer:
    # Callbacks
    callback_list: list[Any] = [NaNCheckCallback()]
    for _, callback_cfg in cfg.callback.callbacks.items():
        callback_list.append(ModelCheckpoint(**callback_cfg, dirpath=ckpt_dir))
    callback_list.append(TQDMProgressBar(refresh_rate=cfg.callback.tqdm_refresh_rate))

    # Other callbacks
    for _, callback_cfg in cfg.callback.other_callbacks.items():
        callback_list.append(instantiate(cfg=callback_cfg))

    # Logger
    pl_logger = TensorBoardLogger(save_dir=cfg.result_dir, name=cfg.name, version=str(cfg.seed))

    # Trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callback_list, logger=pl_logger)
    return trainer


def setup_datamodule(cfg: TrainConfig) -> pl.LightningDataModule:
    logger.info("Create datamodule")
    dm = PLDataModule(dm_cfg=cfg.datamodule)
    return dm


def setup_model(cfg: TrainConfig) -> pl.LightningModule:
    logger.info("Create model")
    model = instantiate(cfg=cfg.plmodel)
    return model


@hydra.main(version_base=None, config_path="../../../config/train", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: TrainConfig = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=TrainConfig)
    if not cfg.trainer.get("deterministic", False):
        logger.warning("Not deterministic!")
        raise ValueError("Not deterministic!")
    else:
        logger.info("Deterministic mode enabled.")
    logger.info(f"Start experiment: {HydraConfig().get().run.dir}")
    pl.seed_everything(cfg.seed, workers=True)

    ckpt_dir = Path(cfg.result_dir) / cfg.name / str(cfg.seed) / "model" / "checkpoints"
    if ckpt_dir.exists():
        logger.warning("Already done. Skipping training...")
        return

    dm = setup_datamodule(cfg=cfg)
    trainer = make_trainer(cfg=cfg, ckpt_dir=ckpt_dir)

    model = setup_model(cfg=cfg)
    model.train()
    logger.info("Start Training")
    trainer.fit(model=model, datamodule=dm, ckpt_path=None)


if __name__ == "__main__":
    main()
