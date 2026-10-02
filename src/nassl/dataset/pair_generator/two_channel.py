# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as functional

from nassl.dataset.utils import stage_paths_to_local_data
from nassl.utils.io import read_json, read_sf

logger = logging.getLogger(__name__)


def _load_rir(rir_path: Path, fs: int, reference_tensor: torch.Tensor) -> torch.Tensor:
    rir_np, sr, _ = read_sf(wav_path=rir_path)
    assert sr == fs
    rir = torch.as_tensor(rir_np, device=reference_tensor.device, dtype=reference_tensor.dtype)
    assert rir.ndim == 1
    return rir


def _load_rir_batch(
    rir_sample_dir_list: List[Path],
    filename: str,
    fs: int,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    rir_list: List[torch.Tensor] = []
    max_len = 0
    for rir_sample_dir in rir_sample_dir_list:
        rir_path = rir_sample_dir / filename
        rir = _load_rir(rir_path=rir_path, fs=fs, reference_tensor=reference_tensor)
        rir_list.append(rir)
        max_len = max(max_len, int(rir.numel()))
    rir_batch = torch.zeros(
        (len(rir_list), max_len),
        device=reference_tensor.device,
        dtype=reference_tensor.dtype,
    )
    for idx, rir in enumerate(rir_list):
        rir_batch[idx, : rir.numel()] = rir
    return rir_batch


def batch_trim_by_delay(full: torch.Tensor, delay: List[int], out_len: int) -> torch.Tensor:
    assert full.ndim == 2

    # Setup
    batch_size, full_len = full.shape
    delay_tensor = torch.tensor(delay, device=full.device, dtype=torch.long)
    assert delay_tensor.shape == (batch_size,)
    if torch.any(delay_tensor < 0):
        raise ValueError("delay must be non-negative")

    # Padding
    max_end = int(delay_tensor.max().item()) + out_len
    if max_end > full_len:
        full = functional.pad(full, (0, max_end - full_len))

    base = torch.arange(out_len, device=full.device).unsqueeze(0)
    indices = delay_tensor.unsqueeze(1) + base

    return torch.gather(full, dim=1, index=indices)


def _convolve_same_batch_conv(wave: torch.Tensor, rir: torch.Tensor, delay: List[int]) -> torch.Tensor:
    assert wave.ndim == 2
    assert rir.ndim == 2
    assert wave.shape[0] == rir.shape[0]
    assert rir.numel() > 0
    batch_size, wave_len = wave.shape
    rir_len = rir.shape[1]
    wave_3d = wave[None, :, :]
    rir_3d = rir.flip(1)[:, None, :]
    full = functional.conv1d(wave_3d, rir_3d, padding=rir_len - 1, groups=batch_size)
    full = full[0]
    return batch_trim_by_delay(full=full, delay=delay, out_len=wave_len)


def _convolve_same_batch_fft(wave: torch.Tensor, rir: torch.Tensor, delay: List[int]) -> torch.Tensor:
    assert wave.ndim == 2
    assert rir.ndim == 2
    assert wave.shape[0] == rir.shape[0]
    assert rir.numel() > 0
    wave_len = wave.shape[1]
    rir_len = rir.shape[1]
    conv_len = int(wave_len + rir_len - 1)
    n_fft = 1 << (conv_len - 1).bit_length()
    wave_f = torch.fft.rfft(wave, n=n_fft)
    rir_f = torch.fft.rfft(rir, n=n_fft)
    full = torch.fft.irfft(wave_f * rir_f, n=n_fft)
    full = full[:, :conv_len]
    return batch_trim_by_delay(full=full, delay=delay, out_len=wave_len)


def _convolve_same_batch(wave: torch.Tensor, rir: torch.Tensor, method: str, delay: List[int]) -> torch.Tensor:
    assert method in ["conv", "fft"]
    if method == "conv":
        return _convolve_same_batch_conv(wave=wave, rir=rir, delay=delay)
    return _convolve_same_batch_fft(wave=wave, rir=rir, delay=delay)


def get_direct_path_delay_list(rir_metadata_list: List[Dict[str, Any]], fs: int) -> List[int]:
    pra_c = 343.0
    pra_frac_delay_length_half = 40
    direct_path_delay_list = []
    for metadata in rir_metadata_list:
        delay = int(metadata["distance_target_ch0"] / pra_c * fs)
        direct_path_delay_list.append(delay + pra_frac_delay_length_half)
    return direct_path_delay_list


def _truncate_rir_batch_by_cutoff(rir: torch.Tensor, cutoff_list: List[int]) -> torch.Tensor:
    assert rir.ndim == 2
    batch_size, rir_len = rir.shape
    cutoff_tensor = torch.tensor(cutoff_list, device=rir.device, dtype=torch.long)
    assert cutoff_tensor.shape == (batch_size,)
    assert bool(torch.all(cutoff_tensor > 0).item())
    sample_index = torch.arange(rir_len, device=rir.device).unsqueeze(0)
    keep_mask = sample_index < cutoff_tensor.unsqueeze(1)
    return rir * keep_mask.to(dtype=rir.dtype)


def generate_early_reflection_target_ch0(
    rir_sample_dir_list: List[Path],
    rir_metadata_list: List[Dict[str, Any]],
    wave_s: torch.Tensor,
    fs: int,
    conv_method: str = "fft",
    early_reflection_duration_sec: float = 0.05,
) -> torch.Tensor:
    assert wave_s.ndim == 2
    assert early_reflection_duration_sec >= 0.0
    direct_path_delay_list = get_direct_path_delay_list(rir_metadata_list=rir_metadata_list, fs=fs)
    early_reflection_len = int(early_reflection_duration_sec * fs)
    early_reflection_cutoff_list = [d + early_reflection_len for d in direct_path_delay_list]
    rir_target_ch0 = _load_rir_batch(
        rir_sample_dir_list=rir_sample_dir_list,
        filename="target_ch0.wav",
        fs=fs,
        reference_tensor=wave_s,
    )
    rir_target_ch0 = _truncate_rir_batch_by_cutoff(rir=rir_target_ch0, cutoff_list=early_reflection_cutoff_list)
    return _convolve_same_batch(
        wave=wave_s,
        rir=rir_target_ch0,
        method=conv_method,
        delay=[0 for _ in rir_metadata_list],
    )


def generate_multi_noise_two_noisy(
    rir_sample_dir_list: List[Path],
    rir_metadata_list: List[Dict[str, Any]],
    wave_s: torch.Tensor,
    wave_n: torch.Tensor,
    snr: torch.Tensor,
    fs: int,
    conv_method: str = "fft",
    remove_direct_path_delay: bool = True,
    num_src: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert wave_s.ndim == 2
    assert wave_n.ndim == 3
    assert wave_s.shape[0] == wave_n.shape[0]
    assert wave_s.shape[1] == wave_n.shape[2]
    assert wave_n.shape[1] == num_src
    assert snr.ndim == 1
    assert wave_s.shape[0] == snr.shape[0]
    if remove_direct_path_delay:
        direct_path_delay_list = get_direct_path_delay_list(rir_metadata_list=rir_metadata_list, fs=fs)
    else:
        direct_path_delay_list = [0 for _ in rir_metadata_list]

    # Generate two-channel clean signal
    rir_target_ch0 = _load_rir_batch(
        rir_sample_dir_list=rir_sample_dir_list,
        filename="target_ch0.wav",
        fs=fs,
        reference_tensor=wave_s,
    )
    wave_s_ch0 = _convolve_same_batch(
        wave=wave_s,
        rir=rir_target_ch0,
        method=conv_method,
        delay=direct_path_delay_list,
    )
    del rir_target_ch0

    rir_target_ch1 = _load_rir_batch(
        rir_sample_dir_list=rir_sample_dir_list,
        filename="target_ch1.wav",
        fs=fs,
        reference_tensor=wave_s,
    )
    wave_s_ch1 = _convolve_same_batch(
        wave=wave_s,
        rir=rir_target_ch1,
        method=conv_method,
        delay=direct_path_delay_list,
    )
    del rir_target_ch1

    # Generate two-channel noise signal
    wave_n_ch0 = torch.zeros_like(wave_s_ch0)
    wave_n_ch1 = torch.zeros_like(wave_s_ch1)
    for src_idx in range(num_src):
        rir_n_ch0 = _load_rir_batch(
            rir_sample_dir_list=rir_sample_dir_list,
            filename=f"noise_corner_{src_idx}_ch0.wav",
            fs=fs,
            reference_tensor=wave_s,
        )
        wave_n_ch0 = wave_n_ch0 + _convolve_same_batch(
            wave=wave_n[:, src_idx, :],
            rir=rir_n_ch0,
            method=conv_method,
            delay=direct_path_delay_list,
        )
        del rir_n_ch0

        rir_n_ch1 = _load_rir_batch(
            rir_sample_dir_list=rir_sample_dir_list,
            filename=f"noise_corner_{src_idx}_ch1.wav",
            fs=fs,
            reference_tensor=wave_s,
        )
        wave_n_ch1 = wave_n_ch1 + _convolve_same_batch(
            wave=wave_n[:, src_idx, :],
            rir=rir_n_ch1,
            method=conv_method,
            delay=direct_path_delay_list,
        )
        del rir_n_ch1

    # Control SNR based on ch0
    eps = 1.0e-8
    power_s_ch0 = torch.mean(wave_s_ch0**2, dim=1)
    power_n_ch0 = torch.mean(wave_n_ch0**2, dim=1)
    snr = snr.to(device=wave_s.device, dtype=wave_s.dtype)
    target_noise_power = power_s_ch0 / (10.0 ** (snr / 10.0))
    scale = torch.sqrt(target_noise_power / (power_n_ch0 + eps))

    # Generate noisy signal
    wave_x_ch0 = wave_s_ch0 + wave_n_ch0 * scale[:, None]
    wave_x_ch1 = wave_s_ch1 + wave_n_ch1 * scale[:, None]
    return wave_x_ch0, wave_x_ch1


def _stage_rir_audio_to_local_cache(rir_dir: Path, local_cache_root: Union[str, None]) -> Path:
    if local_cache_root is None:
        return rir_dir

    rir_audio_dir = rir_dir / "audio"
    logger.info("Staging target_ch*.wav and noise_corner_*.wav RIR files.")
    rir_audio_path_list = sorted(rir_audio_dir.glob("*/target_ch*.wav"))
    rir_audio_path_list += sorted(rir_audio_dir.glob("*/noise_corner_*.wav"))

    staged_path_list = stage_paths_to_local_data(path_list=rir_audio_path_list, local_cache_root=local_cache_root)
    assert len(staged_path_list) > 0
    assert all(staged_path_list[0].parents[2] == p.parents[2] for p in staged_path_list)
    return staged_path_list[0].parents[2]


class MultiNoiseTwoChannelPairGenerator:
    def __init__(
        self,
        snr_range: Tuple[float, float],
        rir_dir: Union[Path, str],
        snr_beta_alpha: float = 1.0,
        snr_beta_beta: float = 1.0,
        fs: int = 16000,
        shuffle: bool = False,
        conv_method: str = "fft",
        local_cache_root: Union[str, None] = None,
        replace_zero_target: bool = False,
        use_early_reflection_target: bool = False,
        early_reflection_duration_sec: float = 0.05,
        return_manifest: bool = False,
        num_src: int = 4,
    ) -> None:
        self.snr_range = snr_range
        self.snr_beta_alpha = snr_beta_alpha
        self.snr_beta_beta = snr_beta_beta
        rir_dir = Path(rir_dir)
        self.rir_metadata_list: List[Dict[str, Any]] = read_json(rir_dir / "metadata_ir.json")  # type: ignore
        self.rir_dir = _stage_rir_audio_to_local_cache(rir_dir=rir_dir, local_cache_root=local_cache_root)
        self.fs = fs
        self.conv_method = conv_method
        self.replace_zero_target = replace_zero_target
        self.use_early_reflection_target = use_early_reflection_target
        self.early_reflection_duration_sec = early_reflection_duration_sec
        self.return_manifest = return_manifest
        assert self.snr_range[0] <= self.snr_range[1]
        assert self.snr_beta_alpha > 0.0
        assert self.snr_beta_beta > 0.0
        assert self.early_reflection_duration_sec >= 0.0
        for i in range(len(self.rir_metadata_list)):
            assert (self.rir_dir / f"audio/{i:06d}").exists()
        assert shuffle
        self.num_src = num_src
        assert self.num_src > 0

    def _generate_beta_tensor1d(
        self, reference_tensor: torch.Tensor, min_value: float, max_value: float
    ) -> torch.Tensor:
        assert min_value <= max_value
        beta_sample_batch = torch.distributions.Beta(
            concentration1=self.snr_beta_alpha,
            concentration0=self.snr_beta_beta,
        ).sample((len(reference_tensor),))
        beta_sample_batch = beta_sample_batch.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )
        random_value_batch = min_value + (max_value - min_value) * beta_sample_batch
        return random_value_batch

    def _check_batch_dict(self, batch_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.replace_zero_target:
            return batch_dict
        wave_s = batch_dict["wave_s"]
        assert wave_s.ndim == 2

        is_zero = torch.all(wave_s == 0, dim=1)
        valid = torch.where(~is_zero)[0]

        if is_zero.any():
            if len(valid) > 0:
                logger.warning(
                    "Found zero target signals in the batch. Replacing them with random samples from non-zero "
                    "target signals."
                )
                batch_dict["wave_s"][is_zero] = wave_s[
                    valid[torch.randint(len(valid), (is_zero.sum(),), device=wave_s.device)]
                ].clone()
            else:
                logger.warning("All target signals in the batch are zero. Replacing them with random noise.")
                batch_dict["wave_s"][is_zero] = torch.randn_like(batch_dict["wave_s"][is_zero])
        return batch_dict

    def _build_pair_manifest(
        self,
        snr_x: torch.Tensor,
        rir_sample_dir_list: List[Path],
        rir_metadata_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        manifest: List[Dict[str, Any]] = []
        for idx, rir_sample_dir in enumerate(rir_sample_dir_list):
            manifest.append(
                {
                    "snr_db": float(snr_x[idx].item()),
                    "rir_sample_dir": str(rir_sample_dir),
                    "rir_metadata": rir_metadata_list[idx],
                }
            )
        return manifest

    def __call__(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert "wave_n_ref" not in batch_dict
        assert "wave_x" not in batch_dict
        batch_dict = self._check_batch_dict(batch_dict=batch_dict)

        wave_s = batch_dict["wave_s"]
        wave_n = batch_dict["wave_n"]
        assert wave_s.ndim == 2
        assert wave_n.ndim == 3
        assert wave_s.shape[0] == wave_n.shape[0]
        assert wave_s.shape[1] == wave_n.shape[2]
        assert wave_n.shape[1] == self.num_src

        snr_x = self._generate_beta_tensor1d(
            reference_tensor=wave_s,
            min_value=self.snr_range[0],
            max_value=self.snr_range[1],
        )
        rir_indices = torch.randint(0, len(self.rir_metadata_list), (wave_s.shape[0],))
        rir_sample_dir_list = [self.rir_dir / f"audio/{int(idx):06d}" for idx in rir_indices]
        rir_metadata_list = [self.rir_metadata_list[int(idx)] for idx in rir_indices]
        wave_x, wave_n_ref = generate_multi_noise_two_noisy(
            rir_sample_dir_list=rir_sample_dir_list,
            rir_metadata_list=rir_metadata_list,
            wave_s=wave_s,
            wave_n=wave_n,
            snr=snr_x,
            fs=self.fs,
            conv_method=self.conv_method,
            remove_direct_path_delay=not self.use_early_reflection_target,
            num_src=self.num_src,
        )
        if self.use_early_reflection_target:
            batch_dict["wave_s"] = generate_early_reflection_target_ch0(
                rir_sample_dir_list=rir_sample_dir_list,
                rir_metadata_list=rir_metadata_list,
                wave_s=wave_s,
                fs=self.fs,
                conv_method=self.conv_method,
                early_reflection_duration_sec=self.early_reflection_duration_sec,
            )

        batch_dict["wave_x"] = wave_x
        batch_dict["wave_n_ref"] = wave_n_ref
        if self.return_manifest:
            batch_dict["pair_manifest"] = self._build_pair_manifest(
                snr_x=snr_x,
                rir_sample_dir_list=rir_sample_dir_list,
                rir_metadata_list=rir_metadata_list,
            )
        return batch_dict
