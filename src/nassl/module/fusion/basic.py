# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import RMSNorm

logger = logging.getLogger(__name__)


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, input_dim)

    def init_identity(self) -> None:
        nn.init.zeros_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PreNormCrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attention_cfg: dict,
        hidden_dim_multiplier: float = 4.0,
        norm_eps: float = 1.0e-6,
        ffn_type: str = "swiglu",
        init_identity: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * hidden_dim_multiplier)
        assert hidden_dim > 0

        self.norm_q = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        self.norm_kv = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        self.attention = nn.MultiheadAttention(**attention_cfg, embed_dim=embed_dim, batch_first=False)
        self.norm_ffn = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(input_dim=embed_dim, hidden_dim=hidden_dim)
        else:
            raise NotImplementedError

        if init_identity:
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)
            self.ffn.init_identity()

        # Calculate number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized PreNormCrossAttnBlock with {num_params} parameters.")

    def forward(
        self,
        x: torch.Tensor,  # T x B x C
        n_ref: torch.Tensor,  # T' x B x C
        padding_mask_n_ref: torch.Tensor | None = None,
        pos_bias=None,
    ) -> torch.Tensor:
        q = self.norm_q(x)
        kv = self.norm_kv(n_ref)
        attn_out, _ = self.attention(
            q,
            kv,
            kv,
            key_padding_mask=padding_mask_n_ref,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FiLM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        init_identity: bool = False,
    ):
        super().__init__()
        self.to_gamma_beta = nn.Linear(embed_dim, embed_dim * 2)
        if init_identity:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(
        self,
        x: torch.Tensor,  # T x B x C
        n_ref: torch.Tensor,  # T' x B x C
        padding_mask_n_ref: torch.Tensor | None = None,
        pos_bias=None,
    ) -> torch.Tensor:
        assert x.dim() == 3 and n_ref.dim() == 3
        assert x.shape[1:] == n_ref.shape[1:]
        assert padding_mask_n_ref is None

        gamma_beta = self.to_gamma_beta(n_ref.mean(dim=0))  # B x C -> B x 2C
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # reshape for broadcasting over spatial / temporal dims
        gamma = gamma.unsqueeze(0)  # 1 x B x C
        beta = beta.unsqueeze(0)  # 1 x B x C

        return gamma * x + beta


class PreNormFiLMFFNBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim_multiplier: float = 4.0,
        norm_eps: float = 1.0e-6,
        ffn_type: str = "swiglu",
        init_identity: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * hidden_dim_multiplier)
        assert hidden_dim > 0

        self.norm_x = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        self.norm_n_ref = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        self.norm_ffn = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(input_dim=embed_dim, hidden_dim=hidden_dim)
        else:
            raise NotImplementedError

        self.film = FiLM(embed_dim=embed_dim, init_identity=init_identity)

        if init_identity:
            self.ffn.init_identity()

        # Calculate number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized PreNormFiLMFFNBlock with {num_params} parameters.")

    def forward(
        self,
        x: torch.Tensor,  # T x B x C
        n_ref: torch.Tensor,  # T' x B x C
        padding_mask_n_ref: torch.Tensor | None = None,
        pos_bias=None,
    ) -> torch.Tensor:
        x = x + self.film(self.norm_x(x), self.norm_n_ref(n_ref), padding_mask_n_ref)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class PreNormSelfAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        attention_cfg: dict,
        hidden_dim_multiplier: float = 4.0,
        norm_eps: float = 1.0e-6,
        ffn_type: str = "swiglu",
        init_identity: bool = False,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * hidden_dim_multiplier)
        assert hidden_dim > 0

        self.norm = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        self.attention = nn.MultiheadAttention(**attention_cfg, embed_dim=embed_dim, batch_first=False)
        self.norm_ffn = RMSNorm(normalized_shape=embed_dim, eps=norm_eps)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(input_dim=embed_dim, hidden_dim=hidden_dim)
        else:
            raise NotImplementedError

        if init_identity:
            nn.init.zeros_(self.attention.out_proj.weight)
            nn.init.zeros_(self.attention.out_proj.bias)
            self.ffn.init_identity()

        # Calculate number of parameters
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"Initialized PreNormSelfAttnBlock with {num_params} parameters.")

    def forward(
        self,
        x: torch.Tensor,  # T x B x C
        padding_mask: torch.Tensor | None = None,
        pos_bias=None,
    ) -> torch.Tensor:
        h = self.norm(x)
        attn_out, _ = self.attention(
            h,
            h,
            h,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x
