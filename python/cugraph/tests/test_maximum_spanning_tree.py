# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cugraph
from cugraph.tests import utils
import rmm
import cudf
import time
import numpy as np

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

print("Networkx version : {} ".format(nx.__version__))


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED_WEIGHTS)
def test_maximum_spanning_tree_nx(graph_file):
    gc.collect()
    # cugraph
    cuG = utils.read_csv_file(graph_file, read_weights_in_sp=True)
    G = cugraph.Graph()
    G.from_cudf_edgelist(cuG, source="0", destination="1", edge_attr="2")
    # Just for getting relevant timing
    G.view_adj_list()
    t1 = time.time()
    cugraph_mst = cugraph.maximum_spanning_tree(G)
    t2 = time.time() - t1
    print("CuGraph time : " + str(t2))

    # Nx
    df = utils.read_csv_for_nx(graph_file, read_weights_in_sp=True)
    Gnx = nx.from_pandas_edgelist(
        df, create_using=nx.Graph(), source="0", target="1", edge_attr="weight"
    )
    t1 = time.time()
    mst_nx = nx.maximum_spanning_tree(Gnx)
    t2 = time.time() - t1
    print("Nx Time : " + str(t2))

    utils.compare_mst(cugraph_mst, mst_nx)


DATASETS_SIZES = [
    100000,
    1000000,
    10000000,
    100000000,
]


@pytest.mark.skip(reason="Skipping large tests")
@pytest.mark.parametrize("graph_size", DATASETS_SIZES)
def test_random_maximum_spanning_tree_nx(graph_size):
    gc.collect()
    rmm.reinitialize(managed_memory=True)
    df = utils.random_edgelist(
        e=graph_size,
        ef=16,
        dtypes={"src": np.int32, "dst": np.int32, "weight": float},
        drop_duplicates=True,
        seed=123456,
    )
    gdf = cudf.from_pandas(df)
    # cugraph
    G = cugraph.Graph()
    G.from_cudf_edgelist(
        gdf, source="src", destination="dst", edge_attr="weight"
    )
    # Just for getting relevant timing
    G.view_adj_list()
    t1 = time.time()
    cugraph.maximum_spanning_tree(G)
    t2 = time.time() - t1
    print("CuGraph time : " + str(t2))

    # Nx
    Gnx = nx.from_pandas_edgelist(
        df,
        create_using=nx.Graph(),
        source="src",
        target="dst",
        edge_attr="weight",
    )
    t1 = time.time()
    nx.maximum_spanning_tree(Gnx)
    t3 = time.time() - t1
    print("Nx Time : " + str(t3))
    print("Speedup: " + str(t3 / t2))
