# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import cudf
import cugraph
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS, karate


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


def cudify(d):
    if d is None:
        return None

    k = np.fromiter(d.keys(), dtype="int32")
    v = np.fromiter(d.values(), dtype="float32")
    cuD = cudf.DataFrame({"vertex": k, "values": v})
    return cuD


def cugraph_call(G, max_iter, tol, alpha, personalization, nstart, pre_vtx_o_wgt):
    # cugraph Pagerank Call
    t1 = time.time()
    df = cugraph.pagerank(
        G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        personalization=personalization,
        precomputed_vertex_out_weight=pre_vtx_o_wgt,
        nstart=nstart,
    )
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    # Sort Pagerank values
    sorted_pr = []

    df = df.sort_values("vertex").reset_index(drop=True)

    pr_scores = df["pagerank"].to_numpy()
    for i, rank in enumerate(pr_scores):
        sorted_pr.append((i, rank))

    return sorted_pr


# need a different function since the Nx version returns a dictionary
def cugraph_nx_call(G, max_iter, tol, alpha, personalization, nstart):
    # cugraph Pagerank Call
    t1 = time.time()
    pr = cugraph.pagerank(
        G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        personalization=personalization,
        nstart=nstart,
    )
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    return pr


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values
def networkx_call(Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx):

    personalization = None
    if personalization_perc != 0:
        personalization = {}
        personalization_count = int((nnz_vtx.size * personalization_perc) / 100.0)
        nnz_vtx = np.random.choice(
            nnz_vtx, min(nnz_vtx.size, personalization_count), replace=False
        )

        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val / sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val

    z = {k: 1.0 / Gnx.number_of_nodes() for k in Gnx.nodes()}

    # Networkx Pagerank Call
    t1 = time.time()

    pr = nx.pagerank(
        Gnx,
        alpha=alpha,
        nstart=z,
        max_iter=max_iter * 2,
        tol=tol * 0.01,
        personalization=personalization,
    )
    t2 = time.time() - t1

    print("Networkx Time : " + str(t2))

    return pr, personalization


# =============================================================================
# Parameters
# =============================================================================
MAX_ITERATIONS = [500]
TOLERANCE = [1.0e-06]
ALPHA = [0.85]
PERSONALIZATION_PERC = [0, 10, 50]
HAS_GUESS = [0, 1]
HAS_PRECOMPUTED = [0, 1]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# FIXME: the default set of datasets includes an asymmetric directed graph
# (email-EU-core.csv), which currently produces different results between
# cugraph and Nx and fails that test. Investigate, resolve, and use
# utils.DATASETS instead.
#
# https://github.com/rapidsai/cugraph/issues/533
#


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
@pytest.mark.parametrize("alpha", ALPHA)
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
@pytest.mark.parametrize("has_precomputed_vertex_out_weight", HAS_PRECOMPUTED)
def test_pagerank(
    graph_file,
    max_iter,
    tol,
    alpha,
    personalization_perc,
    has_guess,
    has_precomputed_vertex_out_weight,
):

    # NetworkX PageRank
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    nnz_vtx = np.unique(M[["0", "1"]])
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )

    networkx_pr, networkx_prsn = networkx_call(
        Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx
    )

    cu_nstart = None
    pre_vtx_o_wgt = None
    if has_guess == 1:
        cu_nstart = cudify(networkx_pr)
        max_iter = 20
    cu_prsn = cudify(networkx_prsn)

    # cuGraph PageRank
    G = graph_file.get_graph(create_using=cugraph.Graph(directed=True))

    if has_precomputed_vertex_out_weight == 1:
        df = G.view_edge_list()[["src", "weights"]]
        pre_vtx_o_wgt = (
            df.groupby(["src"], as_index=False)
            .sum()
            .rename(columns={"src": "vertex", "weights": "sums"})
        )

    cugraph_pr = cugraph_call(
        G, max_iter, tol, alpha, cu_prsn, cu_nstart, pre_vtx_o_wgt
    )

    # Calculating mismatch
    networkx_pr = sorted(networkx_pr.items(), key=lambda x: x[0])
    err = 0
    assert len(cugraph_pr) == len(networkx_pr)
    for i in range(len(cugraph_pr)):
        if (
            abs(cugraph_pr[i][1] - networkx_pr[i][1]) > tol * 1.1
            and cugraph_pr[i][0] == networkx_pr[i][0]
        ):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.01 * len(cugraph_pr))


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
@pytest.mark.parametrize("alpha", ALPHA)
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
def test_pagerank_nx(graph_file, max_iter, tol, alpha, personalization_perc, has_guess):

    # NetworkX PageRank
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    nnz_vtx = np.unique(M[["0", "1"]])
    Gnx = nx.from_pandas_edgelist(M, source="0", target="1", create_using=nx.DiGraph())

    networkx_pr, networkx_prsn = networkx_call(
        Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx
    )

    cu_nstart = None
    if has_guess == 1:
        cu_nstart = cudify(networkx_pr)
        max_iter = 20
    cu_prsn = cudify(networkx_prsn)

    # cuGraph PageRank with Nx Graph
    cugraph_pr = cugraph_nx_call(Gnx, max_iter, tol, alpha, cu_prsn, cu_nstart)

    # Calculating mismatch
    networkx_pr = sorted(networkx_pr.items(), key=lambda x: x[0])
    cugraph_pr = sorted(cugraph_pr.items(), key=lambda x: x[0])
    err = 0
    assert len(cugraph_pr) == len(networkx_pr)

    for i in range(len(cugraph_pr)):
        if (
            abs(cugraph_pr[i][1] - networkx_pr[i][1]) > tol * 1.1
            and cugraph_pr[i][0] == networkx_pr[i][0]
        ):
            err = err + 1
            print(f"{cugraph_pr[i][1]} and {cugraph_pr[i][1]}")
    print("Mismatches:", err)
    assert err < (0.01 * len(cugraph_pr))


@pytest.mark.parametrize("graph_file", DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("tol", TOLERANCE)
@pytest.mark.parametrize("alpha", ALPHA)
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
@pytest.mark.parametrize("has_precomputed_vertex_out_weight", HAS_PRECOMPUTED)
def test_pagerank_multi_column(
    graph_file,
    max_iter,
    tol,
    alpha,
    personalization_perc,
    has_guess,
    has_precomputed_vertex_out_weight,
):

    # NetworkX PageRank
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    nnz_vtx = np.unique(M[["0", "1"]])

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.DiGraph()
    )

    networkx_pr, networkx_prsn = networkx_call(
        Gnx, max_iter, tol, alpha, personalization_perc, nnz_vtx
    )

    cu_nstart = None
    pre_vtx_o_wgt = None
    if has_guess == 1:
        cu_nstart_temp = cudify(networkx_pr)
        max_iter = 100
        cu_nstart = cudf.DataFrame()
        cu_nstart["vertex_0"] = cu_nstart_temp["vertex"]
        cu_nstart["vertex_1"] = cu_nstart["vertex_0"] + 1000
        cu_nstart["values"] = cu_nstart_temp["values"]

    cu_prsn_temp = cudify(networkx_prsn)
    if cu_prsn_temp is not None:
        cu_prsn = cudf.DataFrame()
        cu_prsn["vertex_0"] = cu_prsn_temp["vertex"]
        cu_prsn["vertex_1"] = cu_prsn["vertex_0"] + 1000
        cu_prsn["values"] = cu_prsn_temp["values"]
    else:
        cu_prsn = cu_prsn_temp

    cu_M = cudf.DataFrame()
    cu_M["src_0"] = cudf.Series(M["0"])
    cu_M["dst_0"] = cudf.Series(M["1"])
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000
    cu_M["weights"] = cudf.Series(M["weight"])

    cu_G = cugraph.Graph(directed=True)
    cu_G.from_cudf_edgelist(
        cu_M,
        source=["src_0", "src_1"],
        destination=["dst_0", "dst_1"],
        edge_attr="weights",
        store_transposed=True,
    )

    if has_precomputed_vertex_out_weight == 1:
        df = cu_M[["src_0", "src_1", "weights"]]
        pre_vtx_o_wgt = (
            df.groupby(["src_0", "src_1"], as_index=False)
            .sum()
            .rename(columns={"weights": "sums"})
        )

    df = cugraph.pagerank(
        cu_G,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        personalization=cu_prsn,
        nstart=cu_nstart,
        precomputed_vertex_out_weight=pre_vtx_o_wgt,
    )

    cugraph_pr = []

    df = df.sort_values("0_vertex").reset_index(drop=True)

    pr_scores = df["pagerank"].to_numpy()
    for i, rank in enumerate(pr_scores):
        cugraph_pr.append((i, rank))

    # Calculating mismatch
    networkx_pr = sorted(networkx_pr.items(), key=lambda x: x[0])
    err = 0
    assert len(cugraph_pr) == len(networkx_pr)
    for i in range(len(cugraph_pr)):
        if (
            abs(cugraph_pr[i][1] - networkx_pr[i][1]) > tol * 1.1
            and cugraph_pr[i][0] == networkx_pr[i][0]
        ):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.01 * len(cugraph_pr))


def test_pagerank_invalid_personalization_dtype():
    dataset_path = karate.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    G = cugraph.Graph(directed=True)
    cu_M = cudf.DataFrame()
    cu_M["src"] = cudf.Series(M["0"])
    cu_M["dst"] = cudf.Series(M["1"])

    cu_M["weights"] = cudf.Series(M["weight"])
    G.from_cudf_edgelist(
        cu_M,
        source="src",
        destination="dst",
        edge_attr="weights",
        store_transposed=True,
    )

    personalization_vec = cudf.DataFrame()
    personalization_vec["vertex"] = [17, 26]
    personalization_vec["values"] = [0.5, 0.75]
    warning_msg = (
        "PageRank requires 'personalization' values to match the "
        "graph's 'edge_attr' type. edge_attr type is: "
        "float32 and got 'personalization' values "
        "of type: float64."
    )

    with pytest.warns(UserWarning, match=warning_msg):
        cugraph.pagerank(G, personalization=personalization_vec)


def test_pagerank_transposed_false():
    G = karate.get_graph(create_using=cugraph.Graph(directed=True))
    warning_msg = (
        "Pagerank expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        cugraph.pagerank(G)
