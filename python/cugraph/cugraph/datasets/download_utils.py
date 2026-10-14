# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


class DownloadChecksumError(ValueError):
    """Raised when a downloaded file fails SHA-256 verification."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str) -> None:
    actual = file_sha256(path)
    if actual != expected_sha256.lower():
        raise DownloadChecksumError(
            f"checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
) -> None:
    """
    Download a file from url to dest, optionally verifying its SHA-256 checksum.

    If dest already exists and force is False, skip the download when no checksum
    is configured or when the existing file matches the expected checksum.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and not force:
        if expected_sha256 is None:
            return
        try:
            verify_file_sha256(dest, expected_sha256)
            return
        except DownloadChecksumError:
            dest.unlink()

    urlretrieve(url, str(dest))

    if expected_sha256 is not None:
        verify_file_sha256(dest, expected_sha256)


def safe_extractall(tar: tarfile.TarFile, path: os.PathLike | str) -> None:
    """
    Extract tar members into path while rejecting path traversal entries.
    """
    if sys.version_info >= (3, 12):
        tar.extractall(path, filter="data")
        return

    dest = os.path.abspath(path)
    dest_prefix = dest + os.sep
    for member in tar.getmembers():
        target = os.path.abspath(os.path.join(dest, member.name))
        if target != dest and not target.startswith(dest_prefix):
            raise tarfile.ExtractError(f"unsafe path in tar archive: {member.name!r}")
    tar.extractall(path)
