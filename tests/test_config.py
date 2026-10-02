# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from omegaconf import OmegaConf
from pydantic import BaseModel

from nassl.utils.hydra import hydra_to_pydantic, instantiate


class ExampleConfig(BaseModel):
    seed: int
    name: str


def test_hydra_to_pydantic_validates_config() -> None:
    hydra_cfg = OmegaConf.create({"seed": 0, "name": "test"})

    cfg = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=ExampleConfig)

    assert cfg == ExampleConfig(seed=0, name="test")


def test_project_instantiate_constructs_object() -> None:
    hydra_cfg = OmegaConf.create(
        {
            "_target_": "nassl.loss.loss.MSELoss",
            "reduction": "sum",
        }
    )

    loss = instantiate(cfg=hydra_cfg)

    assert loss.loss_fn.reduction == "sum"
