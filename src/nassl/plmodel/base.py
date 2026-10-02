# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import numbers
from typing import Any, Literal

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


def grad_norm(module: torch.nn.Module) -> float:
    total_sq = 0.0
    with torch.no_grad():
        for p in module.parameters():
            if p.grad is None:
                continue
            n = p.grad.detach().norm(2).item()
            total_sq += n * n
    return total_sq**0.5


class BasePLModel(pl.LightningModule):
    def __init__(
        self,
        optim_cfg: DictConfig,
        lrscheduler_cfg: DictConfig | None = None,
        save_only_trainable: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.optim_cfg = optim_cfg
        self.lrscheduler_cfg = lrscheduler_cfg
        self.save_only_trainable = save_only_trainable
        self.trainable_keys: set[str] | None = None

    def log_loss(
        self,
        loss: torch.Tensor,
        base_name: str,
        log_type: Literal["step", "epoch", "both"] = "step",
        batch_size: int | None = None,
        prog_bar: bool = False,
    ) -> None:
        if loss.ndim != 0:
            raise ValueError(f"Expected scalar loss tensor, got shape={tuple(loss.shape)}")

        if log_type in ("step", "both"):
            self.log(
                f"{base_name}/step",
                loss,
                on_step=True,
                on_epoch=False,
                prog_bar=prog_bar,
                sync_dist=False,  # avoid per-step sync overhead
            )

        if log_type in ("epoch", "both"):
            if batch_size is None:
                raise ValueError("batch_size must be provided when logging epoch metrics")
            self.log(
                f"{base_name}/epoch",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=prog_bar,
                sync_dist=True,
                batch_size=batch_size,
            )

    @staticmethod
    def _to_float(v) -> float | None:
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return float(v.detach().cpu())
            return None
        if isinstance(v, numbers.Number):
            return float(v)
        return None

    def on_validation_epoch_end(self) -> None:
        trainer = self.trainer
        if trainer is None:
            return
        # prevent duplicated logs under DDP / multi-process
        if hasattr(trainer, "is_global_zero") and not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics

        items = []
        for k, v in metrics.items():
            fv = self._to_float(v)
            if fv is None:
                continue
            items.append(f"{k}={fv:.4f}")

        # stable ordering makes logs easier to diff/scan
        items.sort()

        epoch = self.current_epoch + 1  # 1-based for humans
        logger.info(f"Epoch {epoch}, " + ", ".join(items))

    def on_after_backward(self) -> None:
        opt = self.trainer.optimizers[0]
        current_lr = opt.param_groups[0]["lr"]
        gnorm = grad_norm(self)

        self.log("grad/norm", gnorm, on_step=True, on_epoch=False)
        self.log("grad/lr", current_lr, on_step=True, on_epoch=False)
        self.log("grad/step_size", current_lr * gnorm, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = instantiate({"params": self.parameters(), **self.optim_cfg})

        if self.lrscheduler_cfg is None:
            return optimizer

        scheduler = instantiate({"optimizer": optimizer, **self.lrscheduler_cfg})
        lr_scheduler = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def lr_scheduler_step(self, scheduler: Any, metric: Any | None) -> None:
        if scheduler.__class__.__module__.startswith("timm.scheduler."):
            scheduler.step(self.global_step)
        else:
            super().lr_scheduler_step(scheduler=scheduler, metric=metric)

    def _get_trainable_state_dict_keys(self) -> set[str]:
        trainable_keys: set[str] = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_keys.add(name)
        for name, _ in self.named_buffers():
            trainable_keys.add(name)
            logger.info(f"Adding buffer keys: {name}")
        return trainable_keys

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.save_only_trainable:
            if self.trainable_keys is None:
                self.trainable_keys = self._get_trainable_state_dict_keys()
            checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in self.trainable_keys}
