# Copyright (c) 2023-2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os

import cudf
from cugraph.datasets.dataset import set_download_dir
from cugraph.testing.resultset import load_resultset, default_resultset_download_dir

# FIXME: default_resultset_download_dir is an Object of the DefaultDownloadDir class
# that's defined in dataset.py. In resultset.py, we use both the default_download_dir
# object from dataset.py and ANOTHER copy of it that we instantialize locally.. This
# is totally incorrect and should be merged into using a singular object.

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


###############################################################################
# Tests


def test_load_resultset(tmp_path):
    temp_results_path = tmp_path / "tests" / "resultsets"
    temp_results_path.mkdir(parents=True, exist_ok=True)

    assert temp_results_path.exists()

    # FIXME: shouldn't have to use this behavior
    set_download_dir(tmp_path)
    default_resultset_download_dir.path = temp_results_path

    assert "tests" in os.listdir(tmp_path)
    assert "resultsets.tar.gz" not in os.listdir(tmp_path / "tests")
    assert "traversal_mappings.csv" not in os.listdir(tmp_path)

    load_resultset(
        "traversal", "https://data.rapids.ai/cugraph/results/resultsets.tar.gz"
    )
    # reset to default
    set_download_dir(None)

    assert "resultsets.tar.gz" in os.listdir(tmp_path / "tests")
    assert "traversal_mappings.csv" in os.listdir(temp_results_path)


def test_verify_resultset_load(tmp_path):
    # This test is more detailed than test_load_resultset, where for each module,
    # we check that every single resultset file is included along with the
    # corresponding mapping file.
    set_download_dir(tmp_path)
    temp_results_path = tmp_path / "tests" / "resultsets"
    temp_results_path.mkdir(parents=True, exist_ok=True)

    # FIXME: shouldn't have to use this behavior
    set_download_dir(tmp_path)
    default_resultset_download_dir.path = temp_results_path

    load_resultset(
        "traversal", "https://data.rapids.ai/cugraph/results/resultsets.tar.gz"
    )
    # reset to default
    set_download_dir(None)

    resultsets = os.listdir(temp_results_path)
    downloaded_results = cudf.read_csv(
        temp_results_path / "traversal_mappings.csv", sep=" "
    )
    downloaded_uuids = downloaded_results["#UUID"].values
    for resultset_uuid in downloaded_uuids:
        assert str(resultset_uuid) + ".csv" in resultsets
