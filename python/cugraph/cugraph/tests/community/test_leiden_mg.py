# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cugraph
import cugraph.dask as dcg
from cugraph.datasets import karate_asymmetric, karate, dolphins, polbooks
import cudf
from cudf.testing.testing import assert_frame_equal

from test_leiden import assert_valid_partition, host_modularity


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins]
DATASETS_ASYMMETRIC = [karate_asymmetric]


# =============================================================================
# Helper Functions
# =============================================================================


def get_mg_graph(dataset, directed, renumber=True):
    """Returns an MG graph"""
    ddf = dataset.get_dask_edgelist()

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "wgt", renumber=renumber)

    return dg


def sorted_partition(parts):
    return parts.sort_values("vertex").reset_index(drop=True)


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS_ASYMMETRIC)
def test_mg_leiden_with_edgevals_directed_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=True)
    # Directed graphs are not supported by Leiden and a ValueError should be
    # raised
    with pytest.raises(ValueError):
        parts, mod = dcg.leiden(dg)


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
def test_mg_leiden_with_edgevals_undirected_graph(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=False)
    parts, mod = dcg.leiden(dg, random_state=0)
    parts = parts.compute()

    # Leiden cluster IDs are numbered consecutively by decreasing size, every
    # cluster is connected, and the modularity is the exact modularity of the
    # partition (checked on the SG graph of the dataset)
    G = dataset.get_graph()
    assert_valid_partition(G, parts)
    assert mod == pytest.approx(host_modularity(G, parts), abs=1e-10)


@pytest.mark.mg
@pytest.mark.parametrize("dataset", [karate, dolphins, polbooks])
@pytest.mark.parametrize(
    "params",
    [
        {"random_state": 0},
        {"random_state": 7, "resolution": 0.5, "n_iterations": -1},
        {"random_state": 123, "resolution": 2.0, "n_iterations": 1},
    ],
)
def test_mg_leiden_equals_sg(dask_client, dataset, params):
    # Multi-GPU Leiden is bitwise identical to single-GPU Leiden on a graph
    # with the same internal vertex numbering, for any number of workers.
    # A distributed graph is always renumbered internally. The result rows
    # come in internal-id order, which gives the internal numbering: the
    # result partitions are in rank order (dcg.leiden iterates over the
    # workers of dg._plc_graph, which persist_dask_df_equal_parts_per_worker
    # sorted by rank), and the local vertices of every worker are in
    # internal-id order (renumber=False skips the unrenumbering).
    dg = get_mg_graph(dataset, directed=False, renumber=False)
    mg_parts, mg_mod = dcg.leiden(dg, **params)
    mg_parts = mg_parts.compute().reset_index(drop=True)

    # the SG graph on the internal vertex ids of the MG graph
    num_vertices = len(mg_parts)
    to_internal = cudf.Series(
        cudf.Series(range(num_vertices)).values, index=mg_parts["vertex"].values
    )
    edgelist = dataset.get_edgelist()
    edgelist = edgelist.assign(
        src=edgelist["src"].map(to_internal), dst=edgelist["dst"].map(to_internal)
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(edgelist, "src", "dst", "wgt", renumber=False)
    sg_parts, sg_mod = cugraph.leiden(G, **params)
    sg_parts = sorted_partition(sg_parts)

    assert sg_parts["vertex"].to_numpy().tolist() == list(range(num_vertices))
    assert float(mg_mod).hex() == float(sg_mod).hex()
    assert (
        mg_parts["partition"].to_numpy().tolist()
        == sg_parts["partition"].to_numpy().tolist()
    )


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
def test_mg_leiden_deterministic(dask_client, dataset):
    dg = get_mg_graph(dataset, directed=False)
    parts, mod = dcg.leiden(dg, random_state=42)
    parts_2, mod_2 = dcg.leiden(dg, random_state=42)

    assert float(mod_2) == float(mod)
    assert_frame_equal(
        sorted_partition(parts_2.compute()), sorted_partition(parts.compute())
    )


@pytest.mark.mg
def test_mg_leiden_deprecated_and_invalid_parameters(dask_client):
    dg = get_mg_graph(karate, directed=False)
    parts, mod = dcg.leiden(dg, random_state=0)

    # max_iter and theta are accepted (also positionally) but have no effect
    with pytest.warns(FutureWarning) as record:
        parts_2, mod_2 = dcg.leiden(dg, 100, 1.0, 0, 1.0)
    assert len(record) == 2
    assert float(mod_2) == float(mod)
    assert_frame_equal(
        sorted_partition(parts_2.compute()), sorted_partition(parts.compute())
    )

    with pytest.raises(ValueError):
        dcg.leiden(dg, n_iterations=0)
    with pytest.raises(NotImplementedError):
        dcg.leiden(dg, beta=0.5)
