# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from typing import Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from nassl.utils.config_class.train import DMConfig, DMSplitConfig
from nassl.utils.hydra import instantiate

logger = logging.getLogger(__name__)


def set_num_threads_to_one_func(worker_id):
    torch.set_num_threads(1)


class PLDataModule(pl.LightningDataModule):
    def __init__(self, dm_cfg: DMSplitConfig):
        super().__init__()
        self.dm_cfg = dm_cfg

    @staticmethod
    def get_loader(dm_config: Optional[DMConfig]) -> Optional[DataLoader]:

        if dm_config is None:
            return None

        dataset = instantiate(dm_config.dataset)

        if dm_config.batch_sampler is None:
            batch_sampler = None
        else:
            batch_sampler = instantiate({"dataset": dataset, **dm_config.batch_sampler})

        collator = instantiate(dm_config.collator)

        set_num_threads_to_one = dm_config.dataloader.pop("set_num_threads_to_one", False)
        if set_num_threads_to_one:
            worker_init_fn = set_num_threads_to_one_func
        else:
            worker_init_fn = None

        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            worker_init_fn=worker_init_fn,
            **dm_config.dataloader,
        )

    def train_dataloader(self):
        return self.get_loader(dm_config=self.dm_cfg.train)

    def val_dataloader(self):
        if self.dm_cfg.valid is not None:
            # check validation configuration
            if self.dm_cfg.valid.dataloader.get("shuffle", True):
                logger.warning("Validation dataloader set shuffle=True, which is not recommended")
            if self.dm_cfg.valid.batch_sampler is not None:
                logger.warning("Validation batch_sampler is not None, which is not recommended")
            if self.dm_cfg.valid.collator.get("shuffle", True):
                logger.warning("Validation collator set shuffle=True, which is not recommended")

        return self.get_loader(dm_config=self.dm_cfg.valid)

    def test_dataloader(self):
        return None
