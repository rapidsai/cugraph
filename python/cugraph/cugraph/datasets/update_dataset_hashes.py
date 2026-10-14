# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Regenerate SHA-256 checksums for cuGraph dataset metadata and test resultsets.

Usage:
    python -m cugraph.datasets.update_dataset_hashes
    python -m cugraph.datasets.update_dataset_hashes --check
    python -m cugraph.datasets.update_dataset_hashes --dry-run

When hosted dataset files change, run this script and commit the updated
``sha256`` fields in ``metadata/*.yaml`` and ``RESULTSET_SHA256`` in
``cugraph/testing/resultset.py``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

import yaml

from cugraph.datasets.download_utils import file_sha256

METADATA_DIR = Path(__file__).parent / "metadata"
RESULTSET_MODULE = Path(__file__).parent.parent / "testing" / "resultset.py"
RESULTSET_DOWNLOAD_URL = "https://data.rapids.ai/cugraph/results/resultsets.tar.gz"
SHA256_LINE_RE = re.compile(r"^sha256: .*$", re.MULTILINE)
RESULTSET_SHA256_RE = re.compile(r'(RESULTSET_SHA256 = \(\n\s*")([a-f0-9]+)("\n\))')


def _download(url: str, dest: Path, *, use_cache: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if use_cache and dest.is_file():
        print(f"  using cached {dest.name}")
        return
    print(f"  downloading {url}")
    urlretrieve(url, str(dest))


def _update_yaml_sha256(text: str, sha256: str) -> str:
    if SHA256_LINE_RE.search(text):
        return SHA256_LINE_RE.sub(f"sha256: {sha256}", text)

    updated = []
    for line in text.splitlines(keepends=True):
        updated.append(line)
        if line.startswith("url:"):
            updated.append(f"sha256: {sha256}\n")
    return "".join(updated)


def _update_resultset_sha256(text: str, sha256: str) -> str:
    if not RESULTSET_SHA256_RE.search(text):
        raise RuntimeError(
            f"could not find RESULTSET_SHA256 assignment in {RESULTSET_MODULE}"
        )
    return RESULTSET_SHA256_RE.sub(rf"\1{sha256}\3", text)


def _read_resultset_sha256(text: str) -> str | None:
    match = RESULTSET_SHA256_RE.search(text)
    return match.group(2) if match else None


def _iter_dataset_metadata() -> list[tuple[Path, dict]]:
    entries = []
    for yaml_path in sorted(METADATA_DIR.glob("*.yaml")):
        meta = yaml.safe_load(yaml_path.read_text())
        if meta and "url" in meta:
            entries.append((yaml_path, meta))
    return entries


def regenerate_hashes(
    *,
    dry_run: bool = False,
    check_only: bool = False,
    include_resultset: bool = True,
    cache_dir: Path | None = None,
) -> int:
    mismatches: list[str] = []
    updates: list[tuple[Path, str]] = []
    resultset_update: str | None = None

    if cache_dir is None:
        cache_root = Path(tempfile.mkdtemp(prefix="cugraph_dataset_hashes_"))
        cleanup_cache = True
    else:
        cache_root = cache_dir
        cache_root.mkdir(parents=True, exist_ok=True)
        cleanup_cache = False

    use_cache = cache_dir is not None

    try:
        for yaml_path, meta in _iter_dataset_metadata():
            filename = meta["name"] + meta["file_type"]
            url = meta["url"]
            dest = cache_root / filename

            _download(url, dest, use_cache=use_cache)
            sha256 = file_sha256(dest)
            current = meta.get("sha256")
            print(f"{yaml_path.name}: {sha256}")

            if current != sha256:
                message = f"{yaml_path.name}: {current} -> {sha256}"
                if check_only:
                    mismatches.append(message)
                else:
                    updates.append((yaml_path, sha256))

        if include_resultset:
            dest = cache_root / "resultsets.tar.gz"
            _download(RESULTSET_DOWNLOAD_URL, dest, use_cache=use_cache)
            resultset_sha256 = file_sha256(dest)
            current = _read_resultset_sha256(RESULTSET_MODULE.read_text())
            print(f"resultsets.tar.gz: {resultset_sha256}")

            if current != resultset_sha256:
                message = f"resultsets.tar.gz: {current} -> {resultset_sha256}"
                if check_only:
                    mismatches.append(message)
                else:
                    resultset_update = resultset_sha256

        if check_only:
            if mismatches:
                print("\nchecksum mismatches:", file=sys.stderr)
                for message in mismatches:
                    print(f"  {message}", file=sys.stderr)
                return 1
            print("\nall checksums match")
            return 0

        if dry_run:
            print("\ndry run: no files modified")
            return 0

        changed = False
        for yaml_path, sha256 in updates:
            original = yaml_path.read_text()
            yaml_path.write_text(_update_yaml_sha256(original, sha256))
            print(f"updated {yaml_path}")
            changed = True

        if resultset_update is not None:
            original = RESULTSET_MODULE.read_text()
            RESULTSET_MODULE.write_text(
                _update_resultset_sha256(original, resultset_update)
            )
            print(f"updated {RESULTSET_MODULE}")
            changed = True

        if changed:
            print("\nchecksums updated")
        else:
            print("\nno checksum changes needed")

        return 0
    finally:
        if cleanup_cache:
            shutil.rmtree(cache_root, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate SHA-256 checksums for cuGraph dataset downloads."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify existing checksums without modifying files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and print checksums without modifying files.",
    )
    parser.add_argument(
        "--skip-resultset",
        action="store_true",
        help="Only update dataset metadata YAML files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Reuse this directory for downloaded files instead of a temp dir.",
    )
    args = parser.parse_args(argv)

    if args.check and args.dry_run:
        parser.error("--check and --dry-run cannot be used together")

    return regenerate_hashes(
        dry_run=args.dry_run,
        check_only=args.check,
        include_resultset=not args.skip_resultset,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
