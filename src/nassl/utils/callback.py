# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import Callback
from typing_extensions import override

logger = logging.getLogger(__name__)


class NaNCheckCallback(Callback):
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._check_for_nan(outputs):
            logging.warning("NaN detected in training batch, stopping training.")
            trainer.should_stop = True

    @staticmethod
    def _check_for_nan(outputs: Any) -> bool:
        if isinstance(outputs, torch.Tensor):
            return torch.isnan(outputs).any().item()
        elif isinstance(outputs, dict):
            for _, value in outputs.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any().item():
                    return True
        return False


class TrainableOnlyEMACallback(Callback):
    def __init__(
        self,
        decay: float = 0.999,
        update_starting_at_step: int | None = None,
    ) -> None:
        super().__init__()
        assert 0.0 < decay < 1.0
        if update_starting_at_step is not None:
            assert update_starting_at_step >= 0
        self.decay = decay
        self.update_starting_at_step = update_starting_at_step
        self.ema_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.backup_state: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.latest_update_step = 0

    def _named_trainable_parameters(self, pl_module: pl.LightningModule) -> list[tuple[str, torch.nn.Parameter]]:
        return [(name, param) for name, param in pl_module.named_parameters() if param.requires_grad]

    def _get_current_model_state(
        self, pl_module: pl.LightningModule, state_dict: dict[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        current_model_state = {}
        for name, _ in self._named_trainable_parameters(pl_module=pl_module):
            current_model_state[name] = state_dict[name]
        for name, _ in pl_module.named_buffers():
            if name in state_dict:
                current_model_state[name] = state_dict[name]
            else:
                logger.warning(
                    f"Buffer '{name}' not found in checkpoint state_dict. Likely a non-persistent buffer. Skipping it."
                )
        return OrderedDict(current_model_state)

    def _get_state_dict(
        self,
        pl_module: pl.LightningModule,
        current_model_state: OrderedDict[str, torch.Tensor],
    ) -> OrderedDict[str, torch.Tensor]:
        if not self.ema_state:
            return current_model_state

        named_buffers_dict = dict(pl_module.named_buffers())

        state_dict = {}
        for name, param in current_model_state.items():
            if name in self.ema_state:
                state_dict[name] = self.ema_state[name]
            else:
                assert name in named_buffers_dict
                state_dict[name] = param
        return OrderedDict(state_dict)

    def should_update(self, step_idx: int) -> bool:
        if self.update_starting_at_step is None:
            return True
        return step_idx >= self.update_starting_at_step

    @torch.no_grad()
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_state:
            logger.info(f"Moving EMA state to device: {pl_module.device}")
            for name in self.ema_state:
                self.ema_state[name] = self.ema_state[name].to(pl_module.device)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step_idx = trainer.global_step - 1
        if trainer.global_step <= self.latest_update_step:
            # To handle gradient accumulation
            return
        if not self.should_update(step_idx=step_idx):
            return

        if self.ema_state:
            for name, param in self._named_trainable_parameters(pl_module=pl_module):
                self.ema_state[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        else:
            self.ema_state = OrderedDict(
                (name, param.detach().clone()) for name, param in self._named_trainable_parameters(pl_module=pl_module)
            )
        self.latest_update_step = trainer.global_step

    @torch.no_grad()
    def _swap_in_ema_weights(self, pl_module: pl.LightningModule) -> None:
        self.backup_state = OrderedDict()
        param_dict = dict(pl_module.named_parameters())

        for name, ema_param in self.ema_state.items():
            param = param_dict[name]
            self.backup_state[name] = param.detach().clone()
            param.data.copy_(ema_param.data)

    @torch.no_grad()
    def _restore_original_weights(self, pl_module: pl.LightningModule) -> None:
        param_dict = dict(pl_module.named_parameters())

        for name, orig_param in self.backup_state.items():
            param_dict[name].data.copy_(orig_param.data)

        self.backup_state = OrderedDict()

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_state:
            self._swap_in_ema_weights(pl_module=pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.backup_state:
            self._restore_original_weights(pl_module=pl_module)

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_state:
            self._swap_in_ema_weights(pl_module=pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.backup_state:
            self._restore_original_weights(pl_module=pl_module)

    @override
    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint["current_model_state"] = self._get_current_model_state(
            pl_module=pl_module, state_dict=checkpoint["state_dict"]
        )
        checkpoint["ema_state"] = self.ema_state
        checkpoint["state_dict"] = self._get_state_dict(
            pl_module=pl_module, current_model_state=checkpoint["current_model_state"]
        )
        checkpoint["averaging_state"] = {
            "decay": self.decay,
            "update_starting_at_step": self.update_starting_at_step,
            "latest_update_step": self.latest_update_step,
        }

    @override
    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        self.ema_state = checkpoint["ema_state"]

        averaging_state = checkpoint["averaging_state"]
        self.decay = averaging_state["decay"]
        self.update_starting_at_step = averaging_state["update_starting_at_step"]
        self.latest_update_step = averaging_state["latest_update_step"]

        logger.info("Checkpoint loaded successfully, EMA state and averaging state restored.")
