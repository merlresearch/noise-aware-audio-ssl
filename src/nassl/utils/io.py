# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import soundfile as sf


def write_json(json_path: Union[Path, str], data: Union[Dict[str, Any], List[Any]], indent: int = 2) -> None:
    if Path(json_path).exists():
        raise FileExistsError(f"File {json_path} already exists.")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)


def read_json(json_path: Union[Path, str]) -> Union[Dict[str, Any], List[Any]]:
    with open(json_path) as f:
        data = json.load(f)
    return data


def read_sf(wav_path: Union[Path, str]) -> Tuple[Any, int, str]:
    audio, sr = sf.read(str(wav_path))
    subtype = sf.info(str(wav_path)).subtype
    return audio, sr, subtype


def write_sf(wav_path: Union[Path, str], audio: Any, sr: int, subtype: str) -> None:
    sf.write(str(wav_path), audio, sr, subtype=subtype)


def write_txt(txt_path: Union[Path, str], data: str) -> None:
    if Path(txt_path).exists():
        raise FileExistsError(f"File {txt_path} already exists.")
    with open(txt_path, "w") as f:
        f.write(data + "\n")
