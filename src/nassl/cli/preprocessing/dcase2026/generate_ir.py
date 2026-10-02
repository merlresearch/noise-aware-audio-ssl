# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import hydra
import numpy as np
import pyroomacoustics as pra
from omegaconf import DictConfig
from pydantic import BaseModel, model_validator
from tqdm import tqdm

from nassl.utils.hydra import hydra_to_pydantic
from nassl.utils.io import write_json, write_sf

logger = logging.getLogger(__name__)
CONFIG_PATH = "../../../../../config/" + "/".join(Path(__file__).with_suffix("").parts[-3:])


@dataclass(frozen=True)
class RoomGeometry:
    room_dim: np.ndarray
    target_pos: np.ndarray
    mic_pos: np.ndarray
    noise_pos: list[np.ndarray]


@dataclass(frozen=True)
class RirRoom:
    room: Any
    rt60: float


class Config(BaseModel):
    n_ir: int
    name: str
    output_dir: Path
    fs: int = 16000
    seed: int = 0

    room_x: tuple[float, float] = (3.0, 8.0)
    room_y: tuple[float, float] = (3.0, 8.0)
    room_z: tuple[float, float] = (1.5, 3.0)
    rt60: tuple[float, float] = (0.1, 0.40)
    rt60_beta_alpha: float = 1.0
    rt60_beta_beta: float = 1.0
    z_value: float = 1.0

    # Distance from target to mic 0/1
    mic0_dist: tuple[float, float] = (0.00, 0.25)
    mic1_dist: tuple[float, float | None] = (0.25, 1.50)
    mic1_dist_beta_alpha: float = 1.0
    mic1_dist_beta_beta: float = 1.0

    mic_margin: float = 0.2
    noise_margin: float = 0.2
    src_margin: float = 0.5
    src_location_type: Literal["random", "center"] = "random"

    max_mic_direction_trials: int = 100
    max_rt60_trials: int = 100
    max_generation_trials: int = 100

    @model_validator(mode="after")
    def validate_ranges(self) -> Config:
        range_list = [
            self.room_x,
            self.room_y,
            self.room_z,
            self.rt60,
            self.mic0_dist,
        ]
        for value_range in range_list:
            assert value_range[0] <= value_range[1]

        assert self.mic1_dist[0] >= self.mic0_dist[1]
        if self.mic1_dist[1] is not None:
            assert self.mic1_dist[0] <= self.mic1_dist[1]
        assert self.rt60_beta_alpha > 0.0
        assert self.rt60_beta_beta > 0.0
        assert self.mic1_dist_beta_alpha > 0.0
        assert self.mic1_dist_beta_beta > 0.0

        # Ensure z_value is less than the minimum room height
        for margin_name, margin_value in [
            ("mic_margin", self.mic_margin),
            ("noise_margin", self.noise_margin),
        ]:
            if (self.z_value < margin_value) or (self.z_value > self.room_z[0] - margin_value):
                raise ValueError(
                    f"z_value {self.z_value} must be less than the minimum room height "
                    f"({self.room_z[0]} - {margin_name} {margin_value} = {self.room_z[0] - margin_value})."
                )

        # Check margin
        assert self.mic_margin <= self.noise_margin
        assert self.mic_margin <= self.src_margin
        assert self.room_x[0] / 2 >= self.src_margin
        assert self.room_y[0] / 2 >= self.src_margin
        return self


def get_target_source_position(
    location_type: Literal["random", "center"],
    room_dim: np.ndarray,
    margin: float,
    z_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if location_type == "center":
        x = room_dim[0] / 2
        y = room_dim[1] / 2
    else:
        assert location_type == "random"
        x = rng.uniform(margin, room_dim[0] - margin)
        y = rng.uniform(margin, room_dim[1] - margin)
    return np.array([x, y, z_value], dtype=float)


def sample_horizontal_direction(rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi)
    return np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)


def sample_beta_range(
    rng: np.random.Generator,
    value_range: tuple[float, float],
    alpha: float,
    beta: float,
) -> float:
    min_value, max_value = value_range
    beta_sample = rng.beta(alpha, beta)
    return min_value + (max_value - min_value) * beta_sample


def get_max_inside_room_distance(
    anchor_pos: np.ndarray, direction: np.ndarray, room_dim: np.ndarray, margin: float
) -> float:
    max_dist_list: list[float] = []
    for axis in range(2):
        if direction[axis] > 0:
            max_dist = (room_dim[axis] - margin - anchor_pos[axis]) / direction[axis]
            max_dist_list.append(float(max_dist))
        elif direction[axis] < 0:
            max_dist = (margin - anchor_pos[axis]) / direction[axis]
            max_dist_list.append(float(max_dist))

    max_inside_room_distance = min(max_dist_list)
    assert max_inside_room_distance >= 0.0
    return max_inside_room_distance


def get_corner_pos(room_dim: np.ndarray, margin: float, z_value: float) -> list[np.ndarray]:
    return [
        np.array([margin, margin, z_value]),
        np.array([room_dim[0] - margin, margin, z_value]),
        np.array([margin, room_dim[1] - margin, z_value]),
        np.array([room_dim[0] - margin, room_dim[1] - margin, z_value]),
    ]


def sample_mic_position_within_room(
    target_pos: np.ndarray,
    room_dim: np.ndarray,
    margin: float,
    rng: np.random.Generator,
    max_trials: int,
    dist: tuple[float, float | None],
    dist_beta_alpha: float = 1.0,
    dist_beta_beta: float = 1.0,
) -> np.ndarray:
    for _ in range(max_trials):
        direction = sample_horizontal_direction(rng=rng)
        room_max_dist = get_max_inside_room_distance(
            anchor_pos=target_pos, direction=direction, room_dim=room_dim, margin=margin
        )
        if room_max_dist < dist[0]:
            continue

        if dist[1] is None:
            max_dist = room_max_dist
        else:
            max_dist = min(dist[1], room_max_dist)

        mic_dist = sample_beta_range(
            rng=rng,
            value_range=(dist[0], max_dist),
            alpha=dist_beta_alpha,
            beta=dist_beta_beta,
        )
        return target_pos + mic_dist * direction
    raise ValueError(
        f"Failed to sample valid position for mic with target_pos {target_pos} "
        f"and room_dim {room_dim} after {max_trials} trials."
    )


def sample_geometry(cfg: Config, rng: np.random.Generator) -> RoomGeometry:
    # Room dimensions
    room_dim = np.array([rng.uniform(*cfg.room_x), rng.uniform(*cfg.room_y), rng.uniform(*cfg.room_z)])

    # Corner noise source positions
    noise_pos = get_corner_pos(room_dim=room_dim, margin=cfg.noise_margin, z_value=cfg.z_value)

    # Target position
    target_pos = get_target_source_position(
        location_type=cfg.src_location_type,
        room_dim=room_dim,
        margin=cfg.src_margin,
        z_value=cfg.z_value,
        rng=rng,
    )

    # Mic 0 position
    mic0_pos = sample_mic_position_within_room(
        target_pos=target_pos,
        room_dim=room_dim,
        margin=cfg.mic_margin,
        rng=rng,
        max_trials=cfg.max_mic_direction_trials,
        dist=cfg.mic0_dist,
    )

    # Mic 1 position
    mic1_pos = sample_mic_position_within_room(
        target_pos=target_pos,
        room_dim=room_dim,
        margin=cfg.mic_margin,
        rng=rng,
        max_trials=cfg.max_mic_direction_trials,
        dist=cfg.mic1_dist,
        dist_beta_alpha=cfg.mic1_dist_beta_alpha,
        dist_beta_beta=cfg.mic1_dist_beta_beta,
    )

    return RoomGeometry(
        room_dim=room_dim,
        target_pos=target_pos,
        mic_pos=np.stack([mic0_pos, mic1_pos], axis=1),
        noise_pos=noise_pos,
    )


def build_rir_room(
    geometry: RoomGeometry,
    cfg: Config,
    rng: np.random.Generator,
) -> RirRoom:
    for _ in range(cfg.max_rt60_trials):
        try:
            rt60 = sample_beta_range(
                rng=rng,
                value_range=cfg.rt60,
                alpha=cfg.rt60_beta_alpha,
                beta=cfg.rt60_beta_beta,
            )
            absorption, max_order = pra.inverse_sabine(rt60, geometry.room_dim)

            room = pra.ShoeBox(
                geometry.room_dim,
                fs=cfg.fs,
                materials=pra.Material(absorption),
                max_order=min(max_order, 30),
                use_rand_ism=True,
                air_absorption=True,
            )

            room.add_source(geometry.target_pos)
            for noise_pos in geometry.noise_pos:
                room.add_source(noise_pos)

            room.add_microphone_array(pra.MicrophoneArray(geometry.mic_pos, cfg.fs))
            room.compute_rir()
            return RirRoom(room=room, rt60=float(rt60))
        except Exception:
            continue

    raise RuntimeError(f"Failed to build RIR room after {cfg.max_rt60_trials} trials.")


def build_metadata(geometry: RoomGeometry, rt60: float, fs: int) -> dict[str, Any]:
    mic0_dist = np.linalg.norm(geometry.target_pos - geometry.mic_pos[:, 0])
    mic1_dist = np.linalg.norm(geometry.target_pos - geometry.mic_pos[:, 1])

    return {
        "fs": fs,
        "room_dim": geometry.room_dim.tolist(),
        "rt60": float(rt60),
        "target_source": geometry.target_pos.tolist(),
        "mic_ch0": geometry.mic_pos[:, 0].tolist(),
        "mic_ch1": geometry.mic_pos[:, 1].tolist(),
        "noise_sources": [noise_pos.tolist() for noise_pos in geometry.noise_pos],
        "distance_target_ch0": float(mic0_dist),
        "distance_target_ch1": float(mic1_dist),
    }


def write_rirs(room: Any, sample_dir: Path, fs: int) -> None:
    source_names = [
        "target",
        "noise_corner_0",
        "noise_corner_1",
        "noise_corner_2",
        "noise_corner_3",
    ]
    target_ch0 = np.asarray(room.rir[0][0], dtype=np.float32)
    ch0_target_scale = np.max(np.abs(target_ch0))
    assert ch0_target_scale > 0.0
    # Use target ch0 as the common normalization scale for all RIRs.
    for ch in range(2):
        for src_id, name in enumerate(source_names):
            rir = np.asarray(room.rir[ch][src_id], dtype=np.float32)
            rir /= ch0_target_scale
            write_sf(
                wav_path=sample_dir / f"{name}_ch{ch}.wav",
                audio=rir,
                sr=fs,
                subtype="FLOAT",
            )


def generate_rir(idx: int, output_dir: Path, cfg: Config, rng: np.random.Generator) -> dict[str, Any]:
    for _ in range(cfg.max_generation_trials):
        try:
            geometry = sample_geometry(cfg=cfg, rng=rng)
            rir_room = build_rir_room(geometry=geometry, cfg=cfg, rng=rng)
        except Exception:
            continue

        sample_dir = output_dir / f"{idx:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        write_rirs(room=rir_room.room, sample_dir=sample_dir, fs=cfg.fs)
        return build_metadata(geometry=geometry, rt60=rir_room.rt60, fs=cfg.fs)
    raise RuntimeError(f"Failed to generate valid RIR after {cfg.max_generation_trials} trials.")


def _get_output_dir(cfg: Config) -> Path:
    output_dir = cfg.output_dir / cfg.name / "audio"
    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists.")
    return output_dir


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    # Setup
    cfg: Config = hydra_to_pydantic(hydra_cfg=hydra_cfg, cls=Config)
    np.random.seed(cfg.seed)
    pra.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    output_dir = _get_output_dir(cfg)

    # Generate RIRs
    all_meta: list[dict[str, Any]] = []
    for i in tqdm(range(cfg.n_ir)):
        meta = generate_rir(idx=i, output_dir=output_dir, cfg=cfg, rng=rng)
        all_meta.append(meta)

    write_json(json_path=output_dir.parent / "metadata_ir.json", data=all_meta)


if __name__ == "__main__":
    main()
