# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import cupy as cp
import numpy as np
import pandas as pd
import pytest

from pylibcugraph import (
    SGGraph,
    ResourceHandle,
    GraphProperties,
)
from pylibcugraph import leiden
from pylibcugraph.testing import utils


# =============================================================================
# Test data
# =============================================================================

# 6 vertices, every edge stored in both directions
# fmt: off
_srcs = [0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5]
_dsts = [1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4]
_weights = np.asarray(
    [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2],
    dtype=np.float32,
)
# fmt: on

# Leiden is deterministic, so these results are exact (vertex ids are not
# renumbered; clusters are numbered by decreasing size, ties broken by the
# smallest vertex id).
_expected = {
    "weighted": ([0, 0, 0, 1, 1, 1], 0.21596893567080255),
    "unweighted": ([0, 0, 0, 0, 1, 1], 0.125),
}

# karate (file vertex ids), resolution 1, n_iterations 2, random_state 0. The
# algorithm is deterministic (exact integer decisions, fixed-order modularity
# sum), so the call reproduces these values exactly on every GPU
# fmt: off
_karate_partition = [1, 1, 1, 1, 3, 3, 3, 1, 0, 0, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1,
                     0, 1, 0, 2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 0]
_karate_modularity_hex = "0x1.addd53fca2404p-2"
# fmt: on


def create_graph(srcs, dsts, weights=None, vertex_dtype=np.int32):
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_symmetric=True, is_multigraph=False)
    G = SGGraph(
        resource_handle=resource_handle,
        graph_properties=graph_props,
        src_or_offset_array=cp.asarray(srcs, dtype=vertex_dtype),
        dst_or_index_array=cp.asarray(dsts, dtype=vertex_dtype),
        weight_array=None if weights is None else cp.asarray(weights),
        store_transposed=False,
        renumber=False,
        do_expensive_check=False,
    )
    return resource_handle, G


def check_result(result, expected_clusters, expected_modularity):
    vertices, clusters, modularity = result
    assert vertices.get().tolist() == list(range(len(expected_clusters)))
    assert clusters.get().tolist() == expected_clusters
    assert modularity == pytest.approx(expected_modularity, rel=1e-12)


# =============================================================================
# Tests
# =============================================================================
@pytest.mark.parametrize("vertex_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("weight_dtype", [np.float32, np.float64])
def test_sg_leiden(vertex_dtype, weight_dtype):
    # float64 weights with the values of the float32 weights give the
    # identical result
    resource_handle, G = create_graph(
        _srcs, _dsts, _weights.astype(weight_dtype), vertex_dtype
    )
    result = leiden(
        resource_handle,
        0,
        G,
        resolution=1.0,
        do_expensive_check=True,
        n_iterations=2,
        beta=0.0,
    )
    check_result(result, *_expected["weighted"])
    assert result[0].dtype == vertex_dtype
    assert result[1].dtype == vertex_dtype


def test_sg_leiden_unweighted():
    resource_handle, G = create_graph(_srcs, _dsts)
    check_result(leiden(resource_handle, 0, G), *_expected["unweighted"])


@pytest.mark.parametrize("n_iterations", [1, 2, -1])
def test_sg_leiden_karate(n_iterations):
    pdf = pd.read_csv(
        utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv",
        delimiter=" ",
        header=None,
        names=["src", "dst", "weight"],
        dtype={"src": "int32", "dst": "int32", "weight": "float32"},
    )
    resource_handle, G = create_graph(pdf["src"], pdf["dst"], pdf["weight"])
    vertices, clusters, modularity = leiden(
        resource_handle, 0, G, n_iterations=n_iterations
    )
    assert vertices.get().tolist() == list(range(34))
    if n_iterations == 2:
        assert clusters.get().tolist() == _karate_partition
        assert float(modularity).hex() == _karate_modularity_hex


def test_sg_leiden_deterministic():
    resource_handle, G = create_graph(_srcs, _dsts, _weights)
    _, clusters, modularity = leiden(resource_handle, 123, G, resolution=0.5)
    for seed in (123, 123 + 2**32):  # only the low 32 bits of the seed are used
        _, clusters_2, modularity_2 = leiden(resource_handle, seed, G, resolution=0.5)
        assert modularity_2 == modularity
        assert clusters_2.get().tolist() == clusters.get().tolist()


def test_sg_leiden_deprecated_parameters():
    resource_handle, G = create_graph(_srcs, _dsts, _weights)

    # the former positional signature (resource_handle, random_state, graph,
    # max_level, resolution, theta, do_expensive_check) still works; max_level
    # and theta are ignored
    with pytest.warns(FutureWarning) as record:
        result = leiden(resource_handle, 0, G, 100, 1.0, 1.0, False)
    assert len(record) == 2
    assert "max_level" in str(record[0].message)
    assert "theta" in str(record[1].message)
    check_result(result, *_expected["weighted"])

    with pytest.warns(FutureWarning, match="theta"):
        leiden(
            resource_handle=resource_handle,
            random_state=0,
            graph=G,
            resolution=1.0,
            theta=1.0,
            do_expensive_check=False,
        )

    # no warning without the deprecated parameters
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        leiden(resource_handle, 0, G, resolution=1.0)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"n_iterations": 0}, ValueError),
        ({"n_iterations": -2}, ValueError),
        ({"n_iterations": 2**16}, ValueError),
        ({"n_iterations": 2.0}, TypeError),
        ({"n_iterations": True}, TypeError),
        ({"resolution": -1.0}, ValueError),
        ({"resolution": float("inf")}, ValueError),
        ({"resolution": float("nan")}, ValueError),
        ({"beta": -0.5}, ValueError),
        ({"beta": float("nan")}, ValueError),
        ({"beta": 0.01}, NotImplementedError),
    ],
)
def test_sg_leiden_invalid_parameters(kwargs, error):
    resource_handle, G = create_graph(_srcs, _dsts, _weights)
    with pytest.raises(error):
        leiden(resource_handle, 0, G, **kwargs)


@pytest.mark.parametrize("invalid", ["negative", "asymmetric"])
def test_sg_leiden_invalid_graph(invalid):
    weights = _weights.copy()
    if invalid == "negative":
        weights[0] = weights[8] = -0.1  # both directions of edge (0, 1)
    else:
        weights[0] = 0.2  # (0, 1) has weight 0.2, (1, 0) has weight 0.1
    resource_handle, G = create_graph(_srcs, _dsts, weights)
    with pytest.raises(RuntimeError, match="non-negative|symmetric"):
        leiden(resource_handle, 0, G)
