# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import pytest

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate, dolphins, netscience


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [dolphins, karate, netscience]

MIN_WEIGHTS = [0.05, 0.15]

ENSEMBLE_SIZES = [16, 32]

MAX_LEVELS = [10, 20]

RESOLUTIONS = [0.95, 1.0]

THRESHOLDS = [1e-6, 1e-07]

RANDOM_STATES = [0, 42]


# =============================================================================
# Helper Functions
# =============================================================================


def get_mg_graph(dataset, directed):
    """Returns an MG graph"""
    ddf = dataset.get_dask_edgelist()

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "wgt")

    return dg


def golden_call(filename):
    if filename == "dolphins":
        return 0.4962422251701355
    if filename == "karate":
        return 0.38428664207458496
    if filename == "netscience":
        return 0.9279554486274719


# =============================================================================
# Tests
# =============================================================================
# FIXME: Implement more robust tests


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("min_weight", MIN_WEIGHTS)
@pytest.mark.parametrize("ensemble_size", ENSEMBLE_SIZES)
@pytest.mark.parametrize("max_level", MAX_LEVELS)
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("random_state", RANDOM_STATES)
def test_mg_ecg(
    dask_client,
    dataset,
    min_weight,
    ensemble_size,
    max_level,
    threshold,
    resolution,
    random_state,
):
    filename = dataset.metadata["name"]
    dg = get_mg_graph(dataset, directed=False)
    parts, mod = dcg.ecg(
        dg,
        min_weight=min_weight,
        ensemble_size=ensemble_size,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        random_state=random_state,
    )

    filename = dataset.metadata["name"]
    golden_score = golden_call(filename)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert mod > (0.80 * golden_score)

    # print("mod score = ", mod)

    # FIXME: either call Nx with the same dataset and compare results, or
    # hardcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()
