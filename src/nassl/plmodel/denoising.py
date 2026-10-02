# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig

from nassl.feature_extraction_model.base import BaseExtractor
from nassl.utils.hydra import instantiate

from .base import BasePLModel

logger = logging.getLogger(__name__)


class DenoisingPLModel(BasePLModel):
    def __init__(
        self,
        model_cfg: DictConfig,
        loss_cfg: DictConfig,
        optim_cfg: DictConfig,
        lrscheduler_cfg: Optional[DictConfig] = None,
        save_only_trainable: bool = False,
        teacher_model_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(
            optim_cfg=optim_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            save_only_trainable=save_only_trainable,
        )
        self.model = instantiate(model_cfg)
        self.loss = instantiate(loss_cfg)
        if teacher_model_cfg is not None:
            self.teacher_model: BaseExtractor = instantiate(teacher_model_cfg)
            self.teacher_model.requires_grad_(False)
            self.teacher_model.eval()
        else:
            raise ValueError("teacher_model_cfg must be provided")
        self.rep_s_found_flag: Optional[bool] = None

    def train(self, mode: bool = True) -> "DenoisingPLModel":
        super().train(mode=mode)
        self.teacher_model.eval()
        return self

    def wave2loss(self, wave_x: torch.Tensor, rep_s: torch.Tensor) -> Dict[str, Any]:
        rep_s_est = self.model.forward(wave_x)
        loss_dict = {"main": self.loss(ref=rep_s, est=rep_s_est)}
        return loss_dict

    def training_step(self, batch, batch_idx):
        if "rep_s" in batch:
            raise ValueError("rep_s will not be used during training")
        with torch.no_grad():
            rep_s = self.teacher_model(batch["wave_s"])
        loss_dict = self.wave2loss(wave_x=batch["wave_x"], rep_s=rep_s)
        batch_size = len(batch["wave_s"])
        self.log_loss(
            loss=torch.tensor(batch_size).float(),
            base_name="train/batch_size",
            log_type="step",
        )
        for key, val in loss_dict.items():
            self.log_loss(
                loss=val,
                base_name=f"train/{key}",
                log_type="both",
                batch_size=batch_size,
            )
        return loss_dict["main"]

    def validation_step(self, batch, batch_idx):
        if self.rep_s_found_flag is None:
            self.rep_s_found_flag = "rep_s" in batch
            if self.rep_s_found_flag:
                logger.info("rep_s found in batch during validation step")
            else:
                logger.info("rep_s not found in batch during validation step")

        if self.rep_s_found_flag:
            rep_s = batch["rep_s"]
        else:
            with torch.no_grad():
                rep_s = self.teacher_model(batch["wave_s"])
        loss_dict = self.wave2loss(wave_x=batch["wave_x"], rep_s=rep_s)
        batch_size = len(batch["wave_x"])
        self.log_loss(
            loss=torch.tensor(batch_size).float(),
            base_name="valid/batch_size",
            log_type="step",
        )
        for key, val in loss_dict.items():
            self.log_loss(
                loss=val,
                base_name=f"valid/{key}",
                log_type="epoch",
                batch_size=batch_size,
            )
        return loss_dict["main"]
