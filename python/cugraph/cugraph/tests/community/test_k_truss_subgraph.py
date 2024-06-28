# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import pytest
import networkx as nx
import numpy as np

import cugraph
from cugraph.testing import utils
from cugraph.datasets import polbooks, karate_asymmetric
from numba import cuda


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# These ground truth files have been created by running the networkx ktruss
# function on reference graphs. Currently networkx ktruss has an error such
# that nx.k_truss(G,k-2) gives the expected result for running ktruss with
# parameter k. This fix (https://github.com/networkx/networkx/pull/3713) is
# currently in networkx master and will hopefully will make it to a release
# soon.
def ktruss_ground_truth(graph_file):
    G = nx.read_edgelist(str(graph_file), nodetype=int, data=(("weight", float),))
    df = nx.to_pandas_edgelist(G)
    return df


def compare_k_truss(k_truss_cugraph, k, ground_truth_file):
    k_truss_nx = ktruss_ground_truth(ground_truth_file)

    edgelist_df = k_truss_cugraph.view_edge_list()
    src = edgelist_df["src"]
    dst = edgelist_df["dst"]
    wgt = edgelist_df["weight"]
    assert len(edgelist_df) == len(k_truss_nx)
    for i in range(len(src)):
        has_edge = (
            (k_truss_nx["source"] == src[i])
            & (k_truss_nx["target"] == dst[i])
            & np.isclose(k_truss_nx["weight"], wgt[i])
        ).any()
        has_opp_edge = (
            (k_truss_nx["source"] == dst[i])
            & (k_truss_nx["target"] == src[i])
            & np.isclose(k_truss_nx["weight"], wgt[i])
        ).any()
        assert has_edge or has_opp_edge
    return True


__cuda_version = cuda.runtime.get_version()
__unsupported_cuda_version = (11, 4)


# FIXME: remove when ktruss is supported on CUDA 11.4
@pytest.mark.sg
def test_unsupported_cuda_version():
    """
    Ensures the proper exception is raised when ktruss is called in an
    unsupported env, and not when called in a supported env.
    """
    k = 5

    G = polbooks.get_graph(download=True)
    if __cuda_version == __unsupported_cuda_version:
        with pytest.raises(NotImplementedError):
            cugraph.k_truss(G, k)
    else:
        cugraph.k_truss(G, k)


@pytest.mark.sg
@pytest.mark.skipif(
    (__cuda_version == __unsupported_cuda_version),
    reason="skipping on unsupported CUDA " f"{__unsupported_cuda_version} environment.",
)
@pytest.mark.parametrize("_, nx_ground_truth", utils.DATASETS_KTRUSS)
def test_ktruss_subgraph_Graph(_, nx_ground_truth):

    k = 5
    G = polbooks.get_graph(download=True, create_using=cugraph.Graph(directed=False))
    k_subgraph = cugraph.ktruss_subgraph(G, k, use_weights=False)

    compare_k_truss(k_subgraph, k, nx_ground_truth)


@pytest.mark.sg
@pytest.mark.skipif(
    (__cuda_version == __unsupported_cuda_version),
    reason="skipping on unsupported CUDA " f"{__unsupported_cuda_version} environment.",
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_ktruss_subgraph_Graph_nx():
    k = 5
    dataset_path = polbooks.get_path()
    M = utils.read_csv_for_nx(dataset_path, read_weights_in_sp=True)
    G = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    k_subgraph = cugraph.k_truss(G, k)
    k_truss_nx = nx.k_truss(G, k)

    assert nx.is_isomorphic(k_subgraph, k_truss_nx)


@pytest.mark.sg
@pytest.mark.skipif(
    (__cuda_version == __unsupported_cuda_version),
    reason="skipping on unsupported CUDA " f"{__unsupported_cuda_version} environment.",
)
def test_ktruss_subgraph_directed_Graph():
    k = 5
    edgevals = True
    G = karate_asymmetric.get_graph(
        download=True,
        create_using=cugraph.Graph(directed=True),
        ignore_weights=not edgevals,
    )
    with pytest.raises(ValueError):
        cugraph.k_truss(G, k)
