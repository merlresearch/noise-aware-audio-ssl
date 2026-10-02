# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Literal

import torch


class MSELoss(torch.nn.Module):
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean") -> None:
        super().__init__()
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(est, ref)


class EATMSELoss(torch.nn.Module):
    def __init__(
        self,
        num_extra_tokens: int = 1,
        weight_utterance_loss: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self.num_extra_tokens = num_extra_tokens
        self.weight_utterance_loss = weight_utterance_loss
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)

    def forward(self, ref: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref: [B, L, D]
            est: [B, L, D]
        """
        assert ref.ndim == 3
        assert ref.shape == est.shape
        utterance_loss = self.loss_fn(
            est[:, : self.num_extra_tokens, :],
            ref[:, : self.num_extra_tokens, :],
        )
        frame_loss = self.loss_fn(
            est[:, self.num_extra_tokens :, :],
            ref[:, self.num_extra_tokens :, :],
        )
        return frame_loss + self.weight_utterance_loss * utterance_loss
