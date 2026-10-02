# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
from typing import Dict

import pytest

from nassl.utils import download
from nassl.utils.download import BASE_MODEL_LIST, NA_MODEL_LIST, URL_DICT


def test_model_ids_are_unique_and_have_urls() -> None:
    model_ids = BASE_MODEL_LIST + NA_MODEL_LIST
    assert len(model_ids) == len(set(model_ids))
    assert set(URL_DICT) == set(model_ids)


def test_get_cached_model_path_downloads_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count: Dict[str, int] = {"value": 0}

    def fake_download_url_to_file(
        url: str,
        dst: str,
        progress: bool,
    ) -> None:
        del url, progress
        call_count["value"] += 1
        Path(dst).write_bytes(b"checkpoint")

    monkeypatch.setenv("NASSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(download, "download_url_to_file", fake_download_url_to_file)

    first_path = download.get_cached_model_path(model_id="BEATs_iter3")
    second_path = download.get_cached_model_path(model_id="BEATs_iter3")

    assert first_path == tmp_path / "BEATs_iter3.pt"
    assert second_path == first_path
    assert first_path.read_bytes() == b"checkpoint"
    assert call_count["value"] == 1


def test_get_cached_model_path_removes_partial_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download_url_to_file(url: str, dst: str, progress: bool) -> None:
        del url, progress
        Path(dst).write_bytes(b"partial checkpoint")
        raise RuntimeError("download failed")

    monkeypatch.setenv("NASSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(download, "download_url_to_file", fake_download_url_to_file)

    with pytest.raises(RuntimeError, match="download failed"):
        download.get_cached_model_path(model_id="BEATs_iter3")

    assert not (tmp_path / "BEATs_iter3.pt").exists()
    assert not (tmp_path / "BEATs_iter3.pt.tmp").exists()


def test_get_cached_model_path_uses_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count: Dict[str, int] = {"value": 0}

    def fake_download_url_to_file(
        url: str,
        dst: str,
        progress: bool,
    ) -> None:
        del url, progress
        call_count["value"] += 1
        sleep(0.1)
        Path(dst).write_bytes(b"checkpoint")

    monkeypatch.setenv("NASSL_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(download, "download_url_to_file", fake_download_url_to_file)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(download.get_cached_model_path, model_id="dcase2026_nabeats_seed0") for _ in range(2)
        ]
        model_paths = [future.result() for future in futures]

    assert model_paths == [
        tmp_path / "dcase2026_nabeats_seed0.ckpt",
        tmp_path / "dcase2026_nabeats_seed0.ckpt",
    ]
    assert call_count["value"] == 1


def test_get_cached_model_path_rejects_unknown_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NASSL_CACHE_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="Unknown model_id"):
        download.get_cached_model_path(model_id="unknown")
