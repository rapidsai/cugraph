# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

import cugraph
import cupyx
import cudf
from cugraph.testing import UNDIRECTED_DATASETS
from cugraph.datasets import karate_asymmetric


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def cugraph_call(graph_file, edgevals=False, directed=False):
    G = graph_file.get_graph(
        create_using=cugraph.Graph(directed=directed), ignore_weights=not edgevals
    )
    parts, mod = cugraph.louvain(G)

    return parts, mod


# The goal is to just test that the code runs and returns a result.
# The C/C++ test perform a check for accuracy, but the python test
# is not designed to be a 1:1 comparison with the C/C++ test.
@pytest.mark.sg
@pytest.mark.parametrize("graph_file", UNDIRECTED_DATASETS)
def test_louvain(graph_file):
    cu_parts, cu_mod = cugraph_call(graph_file, edgevals=True)

    assert len(cu_parts) > 0
    assert cu_mod > 0.0


@pytest.mark.sg
def test_louvain_directed_graph():
    with pytest.raises(ValueError):
        cugraph_call(karate_asymmetric, edgevals=True, directed=True)


@pytest.mark.sg
@pytest.mark.parametrize("is_weighted", [True, False])
def test_louvain_csr_graph(is_weighted):
    karate = UNDIRECTED_DATASETS[0]
    df = karate.get_edgelist()

    M = cupyx.scipy.sparse.coo_matrix(
        (df["wgt"].to_cupy(), (df["src"].to_cupy(), df["dst"].to_cupy()))
    )
    M = M.tocsr()

    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    weights = cudf.Series(M.data)
    G_csr = cugraph.Graph()
    G_coo = karate.get_graph()

    if not is_weighted:
        weights = None

    G_csr.from_cudf_adjlist(offsets, indices, weights)

    assert G_csr.is_weighted() is is_weighted

    louvain_csr, mod_csr = cugraph.louvain(G_csr)
    louvain_coo, mod_coo = cugraph.louvain(G_coo)
    louvain_csr = louvain_csr.sort_values("vertex").reset_index(drop=True)
    result_louvain = (
        louvain_coo.sort_values("vertex")
        .reset_index(drop=True)
        .rename(columns={"partition": "partition_coo"})
    )
    result_louvain["partition_csr"] = louvain_csr["partition"]

    parition_diffs = result_louvain.query("partition_csr != partition_coo")

    assert len(parition_diffs) == 0
    assert mod_csr == mod_coo
