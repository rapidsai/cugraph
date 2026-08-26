# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cugraph.datasets.update_dataset_hashes import (
    _update_resultset_sha256,
    _update_yaml_sha256,
)


def test_update_yaml_sha256_replaces_existing_line():
    original = "url: https://example.com/a.csv\nsha256: abc\n"
    updated = _update_yaml_sha256(original, "def")
    assert updated == "url: https://example.com/a.csv\nsha256: def\n"


def test_update_yaml_sha256_inserts_after_url():
    original = "name: karate\nurl: https://example.com/a.csv\n"
    updated = _update_yaml_sha256(original, "def")
    assert updated == ("name: karate\nurl: https://example.com/a.csv\nsha256: def\n")


def test_update_resultset_sha256():
    original = (
        "RESULTSET_SHA256 = (\n"
        '    "f170f03167fc6ffef9e227fca77117785ced2794c31d4605834d2c8f76e827ca"\n'
        ")\n"
    )
    updated = _update_resultset_sha256(original, "abc123")
    assert '    "abc123"\n' in updated
