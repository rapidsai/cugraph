# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
import cupyx
import cudf
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS_UNDIRECTED, karate_asymmetric

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import community and import networkx need to be
# relocated in the third-party group once this gets fixed.
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

try:
    import community
except ModuleNotFoundError:
    pytest.exit(
        "community module not found\n"
        "The python-louvain module needs to be installed\n"
        "please run `pip install python-louvain`"
    )


print("Networkx version : {} ".format(nx.__version__))


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


def cugraph_call(graph_file, edgevals=False, directed=False):
    G = graph_file.get_graph(
        create_using=cugraph.Graph(directed=directed), ignore_weights=not edgevals
    )
    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.louvain(G)
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))

    return parts, mod


def networkx_call(M):
    # z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )
    # Networkx louvain Call
    print("Solving... ")
    t1 = time.time()
    parts = community.best_partition(Gnx)
    t2 = time.time() - t1

    print("Networkx Time : " + str(t2))
    return parts


@pytest.mark.parametrize("graph_file", DATASETS_UNDIRECTED)
def test_louvain(graph_file):
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    cu_parts, cu_mod = cugraph_call(graph_file, edgevals=True)
    nx_parts = networkx_call(M)

    # Calculating modularity scores for comparison
    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1", edge_attr="weight", create_using=nx.Graph()
    )

    cu_parts = cu_parts.to_pandas()
    cu_map = dict(zip(cu_parts["vertex"], cu_parts["partition"]))

    assert set(nx_parts.keys()) == set(cu_map.keys())

    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)

    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (0.82 * nx_mod)
    assert abs(cu_mod - cu_mod_nx) < 0.0001


def test_louvain_directed_graph():
    with pytest.raises(ValueError):
        cugraph_call(karate_asymmetric, edgevals=True, directed=True)


@pytest.mark.parametrize("is_weighted", [True, False])
def test_louvain_csr_graph(is_weighted):
    karate = DATASETS_UNDIRECTED[0]
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
