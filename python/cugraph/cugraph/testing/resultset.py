# Copyright (c) 2023, NVIDIA CORPORATION.
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

from tempfile import NamedTemporaryFile
import random

# import pandas as pd
# import cupy as cp
import numpy as np

import cudf
import networkx as nx
import cugraph
from cugraph.experimental.datasets import (
    dolphins,
    netscience,
    karate_disjoint,
    karate,
    polbooks,
)
from cugraph.testing import utils

# from cugraph.experimental import datasets


# =============================================================================
# Parameters
# =============================================================================
SEEDS = [42]

DIRECTED_GRAPH_OPTIONS = [True, False]

DEPTH_LIMITS = [None, 1, 5, 18]

DATASETS = [dolphins, netscience, karate_disjoint]

# DATASETS = [dolphins, netscience, karate_disjoint, karate]

DATASETS_SMALL = [karate, dolphins, polbooks]


# =============================================================================
# tests/traversal/test_bfs.py
# =============================================================================
test_bfs_results = {}
test_bfs_starts = {}

for ds in DATASETS + [karate]:
    for seed in SEEDS:
        for depth_limit in DEPTH_LIMITS:
            for dirctd in DIRECTED_GRAPH_OPTIONS:
                # this does the work of get_cu_graph_nx_results_and_params
                Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=dirctd)

                random.seed(seed)
                start_vertex = random.sample(list(Gnx.nodes()), 1)[0]
                nx_values = nx.single_source_shortest_path_length(
                    Gnx, start_vertex, cutoff=depth_limit
                )

                test_bfs_results[
                    str("{},{},{},{}".format(seed, depth_limit, ds, dirctd))
                ] = nx_values
    test_bfs_starts[str("{},{}".format(seed, ds))] = start_vertex

for dirctd in DIRECTED_GRAPH_OPTIONS:
    Gnx = utils.generate_nx_graph_from_file(karate.get_path(), directed=dirctd)
    result = cugraph.bfs_edges(Gnx, source=7)
    cugraph_df = cudf.from_pandas(result)
    test_bfs_results[str("{},{},{}").format(ds, dirctd, "nonnative-nx")] = cugraph_df


# =============================================================================
# tests/traversal/test_sssp.py
# =============================================================================
test_sssp_results = {}

SOURCES = [1]

for ds in DATASETS_SMALL:
    for source in SOURCES:
        Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=True)
        nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)
        test_sssp_results[str("{},{},ssdpl").format(ds, source)] = nx_paths

        M = utils.read_csv_for_nx(ds.get_path(), read_weights_in_sp=True)
        edge_attr = "weight"
        Gnx = nx.from_pandas_edgelist(
            M,
            source="0",
            target="1",
            edge_attr=edge_attr,
            create_using=nx.DiGraph(),
        )
        test_sssp_results["Gnx,{}".format(ds)] = utils.convert_nx_view_to_dict(Gnx)

        M["weight"] = M["weight"].astype(np.int32)
        Gnx = nx.from_pandas_edgelist(
            M,
            source="0",
            target="1",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )
        test_sssp_results["Gnx,data_type_conversion,{}".format(ds)] = utils.convert_nx_view_to_dict(Gnx)
        test_sssp_results["Gnx_edges,data_type_conversion,{}".format(ds)] = Gnx.edges(
            data=True
        )
        test_sssp_results[
            "nx_paths,data_type_conversion,{}".format(ds)
        ] = nx.single_source_dijkstra_path_length(Gnx, source)

for dirctd in DIRECTED_GRAPH_OPTIONS:
    for source in SOURCES:
        Gnx = utils.generate_nx_graph_from_file(karate.get_path(), directed=dirctd, edgevals=True)
        if dirctd:
            test_sssp_results["nonnative_input,nx.DiGraph,{}".format(source)] = cugraph.sssp(Gnx, source)
        else:
            test_sssp_results["nonnative_input,nx.Graph,{}".format(source)] = cugraph.sssp(Gnx, source)


G = nx.Graph()
G.add_edge(0, 1, other=10)
G.add_edge(1, 2, other=20)
df = cugraph.sssp(G, 0, edge_attr="other")
test_sssp_results["network_edge_attr"] = df


# =============================================================================
# tests/traversal/test_paths.py
# =============================================================================
CONNECTED_GRAPH = """1,5,3
1,4,1
1,2,1
1,6,2
1,7,2
4,5,1
2,3,1
7,6,2
"""

DISCONNECTED_GRAPH = CONNECTED_GRAPH + "8,9,4"

paths = [("1", "1"), ("1", "5"), ("1", "3"), ("1", "6")]
invalid_paths = {
    "connected": [("-1", "1"), ("0", "42")],
    "disconnected": [("1", "10"), ("1", "8")],
}

test_paths_results = {}

# CONNECTED_GRAPH
with NamedTemporaryFile(mode="w+", suffix=".csv") as graph_tf:
    graph_tf.writelines(CONNECTED_GRAPH)
    graph_tf.seek(0)
    Gnx = nx.read_weighted_edgelist(graph_tf.name, delimiter=",")

    graph_tf.writelines(DISCONNECTED_GRAPH)
    graph_tf.seek(0)
    Gnx_DIS = nx.read_weighted_edgelist(graph_tf.name, delimiter=",")

for path in paths:
    nx_path_length = nx.shortest_path_length(
        Gnx, path[0], target=path[1], weight="weight"
    )
    cu_path_length = cugraph.shortest_path_length(Gnx, path[0], target=path[1])
    test_paths_results[
        str("{},{},{},nx").format(path[0], path[1], "connected")
    ] = nx_path_length
    test_paths_results[
        str("{},{},{},cu").format(path[0], path[1], "connected")
    ] = cu_path_length

# INVALID
for graph in ["connected", "disconnected"]:
    if graph == "connected":
        G = Gnx
    else:
        G = Gnx_DIS
    paths = invalid_paths[graph]
    for path in paths:
        try:
            test_paths_results[
                str("{},{},{},invalid").format(path[0], path[1], graph)
            ] = cugraph.shortest_path_length(G, path[0], path[1])
        except ValueError:
            test_paths_results[
                str("{},{},{},invalid").format(path[0], path[1], graph)
            ] = "ValueError"

# test_shortest_path_length_no_target
test_paths_results[str("1,notarget,nx")] = nx.shortest_path_length(
    Gnx_DIS, source="1", weight="weight"
)
test_paths_results[str("1,notarget,cu")] = cugraph.shortest_path_length(Gnx_DIS, "1")


# GETTERS
def get_bfs_results(test_params):
    return test_bfs_results[test_params]


def get_bfs_starts(test_params):
    return test_bfs_starts[test_params]


def get_sssp_results(test_params):
    return test_sssp_results[test_params]


def get_paths_results(test_params):
    return test_paths_results[test_params]
