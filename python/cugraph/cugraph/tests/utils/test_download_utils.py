# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile

import pytest

from cugraph.datasets.download_utils import (
    DownloadChecksumError,
    download_file,
    file_sha256,
    safe_extractall,
    verify_file_sha256,
)


def test_verify_file_sha256(tmp_path):
    data_path = tmp_path / "data.txt"
    data_path.write_text("hello")
    expected = file_sha256(data_path)

    verify_file_sha256(data_path, expected)

    with pytest.raises(DownloadChecksumError):
        verify_file_sha256(data_path, "0" * 64)


def test_download_file_skips_when_cached_and_valid(tmp_path, monkeypatch):
    dest = tmp_path / "karate.csv"
    dest.write_text("cached")
    expected = file_sha256(dest)
    called = {"count": 0}

    def fake_urlretrieve(url, path):
        called["count"] += 1

    monkeypatch.setattr(
        "cugraph.datasets.download_utils.urlretrieve",
        fake_urlretrieve,
    )

    download_file("https://example.com/karate.csv", dest, expected_sha256=expected)

    assert called["count"] == 0


def test_download_file_redownloads_when_checksum_invalid(tmp_path, monkeypatch):
    dest = tmp_path / "karate.csv"
    dest.write_text("stale")
    (tmp_path / "fresh-content").write_text("fresh")
    expected = file_sha256(tmp_path / "fresh-content")
    called = {"count": 0}

    def fake_urlretrieve(url, path):
        called["count"] += 1
        with open(path, "wb") as f:
            f.write(b"fresh")

    monkeypatch.setattr(
        "cugraph.datasets.download_utils.urlretrieve",
        fake_urlretrieve,
    )

    download_file(
        "https://example.com/karate.csv",
        dest,
        expected_sha256=expected,
    )

    assert called["count"] == 1
    assert dest.read_text() == "fresh"


def test_safe_extractall_rejects_path_traversal(tmp_path):
    extract_dir = tmp_path / "resultsets"
    extract_dir.mkdir()

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="../escape.txt")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"evil"))

    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r:gz") as tar:
        with pytest.raises((tarfile.ExtractError, tarfile.OutsideDestinationError)):
            safe_extractall(tar, extract_dir)

    assert not (tmp_path / "escape.txt").exists()


def test_safe_extractall_allows_regular_members(tmp_path):
    extract_dir = tmp_path / "resultsets"
    extract_dir.mkdir()

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="ok.csv")
        info.size = 5
        tar.addfile(info, io.BytesIO(b"1,2,3"))

    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r:gz") as tar:
        safe_extractall(tar, extract_dir)

    assert (extract_dir / "ok.csv").read_text() == "1,2,3"
