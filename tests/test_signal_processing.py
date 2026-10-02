# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Tuple

import torch

from nassl.module.signal_processing import mix_batch_random_snr, mix_batch_snr


def test_mix_batch_snr_reaches_requested_snr() -> None:
    wave_s = torch.tensor([[0.5, -0.5, 0.5, -0.5]])
    wave_n = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    snr = torch.tensor([6.0])

    wave_x = mix_batch_snr(wave_s=wave_s, wave_n=wave_n, snr=snr, clamp=False)
    scaled_noise = wave_x - wave_s
    measured_snr = 10.0 * torch.log10(torch.mean(wave_s**2, dim=1) / torch.mean(scaled_noise**2, dim=1))

    torch.testing.assert_close(measured_snr, snr)


def test_mix_batch_random_snr_is_seeded_and_in_range() -> None:
    torch.manual_seed(0)
    wave_s = torch.full((3, 8), 0.1)
    wave_n = torch.full((3, 8), 0.1)
    snr_range: Tuple[float, float] = (-5.0, 10.0)

    wave_x, snr = mix_batch_random_snr(wave_s=wave_s, wave_n=wave_n, snr_range=snr_range)

    assert wave_x.shape == wave_s.shape
    assert snr.shape == (wave_s.shape[0],)
    assert torch.all(snr >= snr_range[0])
    assert torch.all(snr <= snr_range[1])
