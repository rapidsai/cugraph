# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import gc

import pytest

import cugraph
import cugraph.dask as dcg
from cudf.testing.testing import assert_frame_equal
from cugraph.datasets import karate, dolphins, netscience


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# =============================================================================
# Parameters
# =============================================================================


DATASETS = [karate, dolphins, netscience]
IS_DIRECTED = [True, False]
K_VALUE = [4, 6, 8]


# =============================================================================
# Helper functions
# =============================================================================


def get_sg_graph(dataset, directed):
    G = dataset.get_graph(create_using=cugraph.Graph(directed=directed))

    return G


def get_mg_graph(dataset, directed):
    ddf = dataset.get_dask_edgelist()
    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        edge_attr="wgt",
        renumber=True,
        store_transposed=True,
    )

    return dg


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.mg
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("is_directed", IS_DIRECTED)
@pytest.mark.parametrize("k", K_VALUE)
def test_mg_ktruss_subgraph(dask_client, benchmark, dataset, is_directed, k):
    # Create SG and MG Graphs
    g = get_sg_graph(dataset, is_directed)
    dg = get_mg_graph(dataset, is_directed)

    if is_directed:
        with pytest.raises(ValueError):
            result_ktruss_subgraph = benchmark(dcg.ktruss_subgraph, dg, k)
    else:
        sg_ktruss_subgraph = cugraph.ktruss_subgraph(g, k=k)
        result_ktruss_subgraph = benchmark(dcg.ktruss_subgraph, dg, k)

        mg_df = result_ktruss_subgraph

        if len(mg_df) != 0 and len(sg_ktruss_subgraph.input_df) != 0:
            # FIXME: 'edges()' or 'view_edgelist()' takes half the edges out if
            # 'directed=False'.
            sg_result = sg_ktruss_subgraph.input_df

            sg_df = sg_result.sort_values(["src", "dst"]).reset_index(drop=True)
            mg_df = mg_df.compute().sort_values(["src", "dst"]).reset_index(drop=True)

            assert_frame_equal(sg_df, mg_df, check_dtype=False, check_like=True)

        else:
            # There is no edge left when extracting the K-Truss
            assert len(sg_ktruss_subgraph.input_df) == 0
            assert len(mg_df) == 0
