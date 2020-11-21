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
import time

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


def cugraph_call(cu_M, edgevals=False):
    G = cugraph.Graph()
    if edgevals is True:
        G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")
    else:
        G.from_cudf_edgelist(cu_M, source="0", destination="1")

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.jaccard(G)
    t2 = time.time() - t1
    print("Time : " + str(t2))

    df = df.sort_values(["source", "destination"]).reset_index(drop=True)

    return (
        df["source"].to_array(),
        df["destination"].to_array(),
        df["jaccard_coeff"].to_array(),
    )


def networkx_call(M):

    sources = M["0"]
    destinations = M["1"]
    edges = []
    for i in range(len(M)):
        edges.append((sources[i], destinations[i]))
        edges.append((destinations[i], sources[i]))
    edges = list(dict.fromkeys(edges))
    edges = sorted(edges)
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this
    # explicitly
    print("Format conversion ... ")

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    # Networkx Jaccard Call
    print("Solving... ")
    t1 = time.time()
    preds = nx.jaccard_coefficient(Gnx, edges)
    t2 = time.time() - t1

    print("Time : " + str(t2))
    src = []
    dst = []
    coeff = []
    for u, v, p in preds:
        src.append(u)
        dst.append(v)
        coeff.append(p)
    return src, dst, coeff


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_jaccard(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_src, cu_dst, cu_coeff = cugraph_call(cu_M)
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


@pytest.mark.parametrize("graph_file", ["../datasets/netscience.csv"])
def test_jaccard_edgevals(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)
    cu_src, cu_dst, cu_coeff = cugraph_call(cu_M, edgevals=True)
    nx_src, nx_dst, nx_coeff = networkx_call(M)

    # Calculating mismatch
    err = 0
    tol = 1.0e-06

    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if abs(cu_coeff[i] - nx_coeff[i]) > tol * 1.1:
            err += 1

    print("Mismatches:  %d" % err)
    assert err == 0


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_jaccard_two_hop(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1")
    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )
    nx_pairs = []
    for i in range(len(pairs)):
        nx_pairs.append((pairs["first"].iloc[i], pairs["second"].iloc[i]))
    preds = nx.jaccard_coefficient(Gnx, nx_pairs)
    nx_coeff = []
    for u, v, p in preds:
        nx_coeff.append(p)
    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=["source", "destination"]).reset_index(drop=True)
    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df["jaccard_coeff"].iloc[i])
        assert diff < 1.0e-6


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_jaccard_two_hop_edge_vals(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source="0", destination="1", edge_attr="2")

    pairs = (
        G.get_two_hop_neighbors()
        .sort_values(["first", "second"])
        .reset_index(drop=True)
    )

    nx_pairs = []
    for i in range(len(pairs)):
        nx_pairs.append((pairs["first"].iloc[i], pairs["second"].iloc[i]))
    preds = nx.jaccard_coefficient(Gnx, nx_pairs)
    nx_coeff = []
    for u, v, p in preds:
        nx_coeff.append(p)
    df = cugraph.jaccard(G, pairs)
    df = df.sort_values(by=["source", "destination"]).reset_index(drop=True)
    assert len(nx_coeff) == len(df)
    for i in range(len(df)):
        diff = abs(nx_coeff[i] - df["jaccard_coeff"].iloc[i])
        assert diff < 1.0e-6


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_jaccard_nx(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", create_using=nx.Graph()
    )

    nx_j = nx.jaccard_coefficient(Gnx)
    nv_js = sorted(nx_j, key=len, reverse=True)

    cg_j = cugraph.jaccard_coefficient(Gnx)

    assert len(nv_js) > len(cg_j)

    # FIXME:  Nx does a full all-pair Jaccard.
    # cuGraph does a limited 1-hop Jaccard
    # assert nx_j == cg_j
