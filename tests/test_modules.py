# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict

import torch

from nassl.loss.loss import EATMSELoss, MSELoss
from nassl.module.fusion.basic import PreNormFiLMFFNBlock, PreNormSelfAttnBlock


def test_self_attention_block_forward_shape() -> None:
    torch.manual_seed(0)
    attention_cfg: Dict[str, object] = {"num_heads": 2, "dropout": 0.0}
    model = PreNormSelfAttnBlock(embed_dim=8, attention_cfg=attention_cfg)
    x = torch.randn(5, 2, 8)

    output = model(x=x)

    assert output.shape == x.shape


def test_film_block_identity_initialization() -> None:
    torch.manual_seed(0)
    model = PreNormFiLMFFNBlock(embed_dim=8, init_identity=True)
    x = torch.randn(5, 2, 8)
    n_ref = torch.randn(3, 2, 8)

    output = model(x=x, n_ref=n_ref)

    torch.testing.assert_close(output, x)


def test_mse_losses_return_scalars() -> None:
    ref = torch.zeros(2, 4, 3)
    est = torch.ones(2, 4, 3)

    mse = MSELoss()(ref=ref, est=est)
    eat_mse = EATMSELoss(num_extra_tokens=1)(ref=ref, est=est)

    assert mse.ndim == 0
    assert eat_mse.ndim == 0
    torch.testing.assert_close(mse, torch.tensor(1.0))
    torch.testing.assert_close(eat_mse, torch.tensor(2.0))
