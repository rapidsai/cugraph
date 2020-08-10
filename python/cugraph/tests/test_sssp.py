# Copyright (c) 2019, NVIDIA CORPORATION.
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
import time

import numpy as np
import pytest

import cugraph
from cugraph.tests import utils

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


def cugraph_call(cu_M, source, edgevals=False):

    G = cugraph.DiGraph()
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")
    print("sources size = " + str(len(cu_M["0"])))
    print("destinations size = " + str(len(cu_M["1"])))

    print("cugraph Solving... ")
    t1 = time.time()

    df = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    if np.issubdtype(df["distance"].dtype, np.integer):
        max_val = np.iinfo(df["distance"].dtype).max
    else:
        max_val = np.finfo(df["distance"].dtype).max

    verts_np = df["vertex"].to_array()
    dist_np = df["distance"].to_array()
    pred_np = df["predecessor"].to_array()
    result = dict(zip(verts_np, zip(dist_np, pred_np)))
    return result, max_val


def networkx_call(M, source, edgevals=False):

    # Directed NetworkX graph
    Gnx = nx.from_pandas_edgelist(
        M,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    print("NX Solving... ")
    t1 = time.time()

    if edgevals is False:
        path = nx.single_source_shortest_path_length(Gnx, source)
    else:
        path = nx.single_source_dijkstra_path_length(Gnx, source)

    t2 = time.time() - t1

    print("NX Time : " + str(t2))

    return path, Gnx


SOURCES = [1]


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS_4)
@pytest.mark.parametrize("source", SOURCES)
def test_sssp(graph_file, source):
    print("DOING test_sssp : " + graph_file + "\n\n\n")
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_paths, max_val = cugraph_call(cu_M, source)
    nx_paths, Gnx = networkx_call(M, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != nx_paths[vid]:
                err = err + 1
            # check pred dist + 1 = current dist (since unweighted)
            pred = cu_paths[vid][1]
            if vid != source and cu_paths[pred][0] + 1 != cu_paths[vid][0]:
                err = err + 1
        else:
            if vid in nx_paths.keys():
                err = err + 1

    assert err == 0
    print("DONE test_sssp : " + graph_file + "\n\n\n")


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS_1)
@pytest.mark.parametrize("source", SOURCES)
def test_sssp_edgevals(graph_file, source):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_paths, max_val = cugraph_call(cu_M, source, edgevals=True)
    nx_paths, Gnx = networkx_call(M, source, edgevals=True)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != nx_paths[vid]:
                err = err + 1
            # check pred dist + edge_weight = current dist
            if vid != source:
                pred = cu_paths[vid][1]
                edge_weight = Gnx[pred][vid]["weight"]
                if cu_paths[pred][0] + edge_weight != cu_paths[vid][0]:
                    err = err + 1
        else:
            if vid in nx_paths.keys():
                err = err + 1

    assert err == 0


@pytest.mark.parametrize("graph_file", utils.DATASETS_1)
@pytest.mark.parametrize("source", SOURCES)
def test_sssp_data_type_conversion(graph_file, source):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    # cugraph call with int32 weights
    cu_M["2"] = cu_M["2"].astype(np.int32)
    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    # assert cugraph weights is int32
    assert G.edgelist.edgelist_df["weights"].dtype == np.int32
    df = cugraph.sssp(G, source)
    max_val = np.finfo(df["distance"].dtype).max
    verts_np = df["vertex"].to_array()
    dist_np = df["distance"].to_array()
    pred_np = df["predecessor"].to_array()
    cu_paths = dict(zip(verts_np, zip(dist_np, pred_np)))

    # networkx call with int32 weights
    M["weight"] = M["weight"].astype(np.int32)
    Gnx = nx.from_pandas_edgelist(
        M,
        source="0",
        target="1",
        edge_attr="weight",
        create_using=nx.DiGraph(),
    )
    # assert nx weights is int
    assert type(list(Gnx.edges(data=True))[0][2]["weight"]) is int
    nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)

    # Calculating mismatch
    err = 0
    for vid in cu_paths:
        # Validate vertices that are reachable
        # NOTE : If distance type is float64 then cu_paths[vid][0]
        # should be compared against np.finfo(np.float64).max)
        if cu_paths[vid][0] != max_val:
            if cu_paths[vid][0] != nx_paths[vid]:
                err = err + 1
            # check pred dist + edge_weight = current dist
            if vid != source:
                pred = cu_paths[vid][1]
                edge_weight = Gnx[pred][vid]["weight"]
                if cu_paths[pred][0] + edge_weight != cu_paths[vid][0]:
                    err = err + 1
        else:
            if vid in nx_paths.keys():
                err = err + 1

    assert err == 0
