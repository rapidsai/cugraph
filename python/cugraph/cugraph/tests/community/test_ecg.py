# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

import cugraph
from cugraph.datasets import karate, dolphins, netscience


def cugraph_call(
    G, min_weight, ensemble_size, max_level, threshold, resolution, random_state
):
    parts, mod = cugraph.ecg(
        G,
        min_weight=min_weight,
        ensemble_size=ensemble_size,
        max_level=max_level,
        threshold=threshold,
        resolution=resolution,
        random_state=random_state,
    )
    num_parts = parts["partition"].max() + 1
    return mod, num_parts


def golden_call(filename):
    if filename == "dolphins":
        return 0.4962422251701355
    if filename == "karate":
        return 0.38428664207458496
    if filename == "netscience":
        return 0.9279554486274719


DATASETS = [karate, dolphins, netscience]

MIN_WEIGHTS = [0.05, 0.15]

ENSEMBLE_SIZES = [16, 32]

MAX_LEVELS = [10, 20]

RESOLUTIONS = [0.95, 1.0]

THRESHOLDS = [1e-6, 1e-07]

RANDOM_STATES = [0, 42]


@pytest.mark.sg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("min_weight", MIN_WEIGHTS)
@pytest.mark.parametrize("ensemble_size", ENSEMBLE_SIZES)
@pytest.mark.parametrize("max_level", MAX_LEVELS)
@pytest.mark.parametrize("threshold", THRESHOLDS)
@pytest.mark.parametrize("resolution", RESOLUTIONS)
@pytest.mark.parametrize("random_state", RANDOM_STATES)
def test_ecg_clustering(
    dataset, min_weight, ensemble_size, max_level, threshold, resolution, random_state
):
    gc.collect()

    # Read in the graph and get a cugraph object
    G = dataset.get_graph()
    # read_weights_in_sp=False => value column dtype is float64
    G.edgelist.edgelist_df["weights"] = G.edgelist.edgelist_df["weights"].astype(
        "float64"
    )

    # Get the modularity score for partitioning versus random assignment
    cu_score, num_parts = cugraph_call(
        G, min_weight, ensemble_size, max_level, threshold, resolution, random_state
    )
    filename = dataset.metadata["name"]
    golden_score = golden_call(filename)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > (0.80 * golden_score)
