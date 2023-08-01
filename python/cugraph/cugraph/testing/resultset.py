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
from pathlib import Path

import numpy as np
import networkx as nx

import cudf
import cugraph
from cugraph.experimental.datasets import (
    dolphins,
    netscience,
    karate_disjoint,
    karate,
    polbooks,
)
from cugraph.testing import utils

results_dir = Path("testing/results")

_resultsets = {}


def add_resultset(result_data_dictionary, **kwargs):
    rs = utils.ResultSet(result_data_dictionary)
    hashable_dict_repr = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    _resultsets[hashable_dict_repr] = rs


def get_resultset(category, **kwargs):
    hashable_dict_repr = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    mappings_path = results_dir / (category + "_mappings.csv")
    # Had to set dtype=str to prevent 1s being converted to True
    mappings = cudf.read_csv(mappings_path, sep=" ", dtype=str)
    colnames = mappings.columns
    query_cols = [t for t in colnames][1:]
    dict_repr = dict(hashable_dict_repr)
    argnames, argvals = [t for t in dict_repr.keys()], [t for t in dict_repr.values()]
    mapping_length = 2 * len(argvals) - 1
    single_mapping = np.empty(mapping_length, dtype=object)
    single_mapping[0] = argvals[0]
    for i in np.arange(1, len(argvals)):
        single_mapping[2 * i - 1] = argnames[i]
        single_mapping[2 * i] = argvals[i]
    for i in np.arange(mapping_length):
        mappings = mappings[mappings[query_cols[i]] == single_mapping[i]]
    # values_host is used instead of values bc strings aren't saved/possible on device
    results_filename = category + "-" + mappings.head(1)["UUID"].values_host[0]
    # results_filename = mappings.head(1)["UUID"].values_host[0]
    results_filename = results_filename + ".csv"
    # Ignore for now -> Assumption is the filename already has the alg category
    path = results_dir / results_filename
    # path = Path("https://data.rapids.ai/cugraph/results/" / path
    return cudf.read_csv(path)


# =============================================================================
# Parameters
# =============================================================================
# This will be refactored once the datasets variables are fixed/changed
SEEDS = [42]

DIRECTED_GRAPH_OPTIONS = [True, False]

DEPTH_LIMITS = [None, 1, 5, 18]

DATASETS = [dolphins, netscience, karate_disjoint]

DATASETS_SMALL = [karate, dolphins, polbooks]

# =============================================================================
# tests/traversal/test_bfs.py
# =============================================================================
test_bfs_results = {}

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
                """test_bfs_results[
                    "{},{},{},{},{}".format(seed, depth_limit, ds, dirctd, start_vertex)
                ] = nx_values"""
                vertices = cudf.Series(nx_values.keys())
                distances = cudf.Series(nx_values.values())
                add_resultset(
                    {"vertex": vertices, "distance": distances},
                    graph_dataset=ds.metadata["name"],
                    graph_directed=str(dirctd),
                    algo="single_source_shortest_path_length",
                    start_vertex=str(start_vertex),
                    cutoff=str(depth_limit),
                )
    # test_bfs_results["{},{},starts".format(seed, ds)] = start_vertex

# these are pandas dataframes
for dirctd in DIRECTED_GRAPH_OPTIONS:
    Gnx = utils.generate_nx_graph_from_file(karate.get_path(), directed=dirctd)
    result = cugraph.bfs_edges(Gnx, source=7)
    cugraph_df = cudf.from_pandas(result)
    # test_bfs_results["{},{},{}".format(ds, dirctd, "nonnative-nx")] = cugraph_df
    add_resultset(
        cugraph_df,
        graph_dataset="karate",
        graph_directed=str(dirctd),
        algo="bfs_edges",
        source="7",
    )


# =============================================================================
# tests/traversal/test_sssp.py
# =============================================================================
test_sssp_results = {}

SOURCES = [1]

for ds in DATASETS_SMALL:
    for source in SOURCES:
        Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=True)
        nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)
        # test_sssp_results["{},{},ssdpl".format(ds, source)] = nx_paths
        vertices = cudf.Series(nx_paths.keys())
        distances = cudf.Series(nx_paths.values())
        add_resultset(
            {"vertex": vertices, "distance": distances},
            graph_dataset=ds.metadata["name"],
            graph_directed="True",
            algo="single_source_dijkstra_path_length",
            source=str(source),
        )

        M = utils.read_csv_for_nx(ds.get_path(), read_weights_in_sp=True)
        edge_attr = "weight"
        Gnx = nx.from_pandas_edgelist(
            M,
            source="0",
            target="1",
            edge_attr=edge_attr,
            create_using=nx.DiGraph(),
        )

        M["weight"] = M["weight"].astype(np.int32)
        Gnx = nx.from_pandas_edgelist(
            M,
            source="0",
            target="1",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )
        nx_paths_datatypeconv = nx.single_source_dijkstra_path_length(Gnx, source)
        """test_sssp_results[
            "nx_paths,data_type_conversion,{}".format(ds)
        ] = nx_paths_datatypeconv"""
        vertices_datatypeconv = cudf.Series(nx_paths_datatypeconv.keys())
        distances_datatypeconv = cudf.Series(nx_paths_datatypeconv.values())
        add_resultset(
            {"vertex": vertices_datatypeconv, "distance": distances_datatypeconv},
            graph_dataset=ds.metadata["name"],
            graph_directed="True",
            algo="single_source_dijkstra_path_length",
            test="data_type_conversion",
            source=str(source),
        )

for dirctd in DIRECTED_GRAPH_OPTIONS:
    for source in SOURCES:
        Gnx = utils.generate_nx_graph_from_file(
            karate.get_path(), directed=dirctd, edgevals=True
        )
        """if dirctd:
            test_sssp_results[
                "nonnative_input,nx.DiGraph,{}".format(source)
            ] = cugraph.sssp(Gnx, source)
        else:
            test_sssp_results[
                "nonnative_input,nx.Graph,{}".format(source)
            ] = cugraph.sssp(Gnx, source)"""
        add_resultset(
            cugraph.sssp(Gnx, source),
            graph_dataset="karate",
            graph_directed=str(dirctd),
            algo="sssp_nonnative",
            source=str(source),
        )

G = nx.Graph()
G.add_edge(0, 1, other=10)
G.add_edge(1, 2, other=20)
df = cugraph.sssp(G, 0, edge_attr="other")
# test_sssp_results["network_edge_attr"] = df
add_resultset(df, algo="sssp_nonnative", test="network_edge_attr")

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

# CONNECTED_GRAPH
with NamedTemporaryFile(mode="w+", suffix=".csv") as graph_tf:
    graph_tf.writelines(DISCONNECTED_GRAPH)
    graph_tf.seek(0)
    Gnx_DIS = nx.read_weighted_edgelist(graph_tf.name, delimiter=",")

res1 = nx.shortest_path_length(Gnx_DIS, source="1", weight="weight")
vertices = cudf.Series(res1.keys())
distances = cudf.Series(res1.values())
add_resultset(
    {"vertex": vertices, "distance": distances},
    algo="shortest_path_length",
    graph_dataset="DISCONNECTED",
    graph_directed="True",
    source="1",
    weight="weight",
)


# Generating ALL results files
"""random.seed(24)
for temp in _resultsets:
    res = _resultsets[temp].get_cudf_dataframe()
    # Currently, only traversal results files are generated
    temp_filename = "traversal-" + str(random.getrandbits(55)) + ".csv"
    temp_mapping = cudf.DataFrame(
        [[str(temp), temp_filename]], columns=["hashable_dict_repr", "filename"]
    )
    traversal_mappings = cudf.concat(
        [traversal_mappings, temp_mapping], axis=0, ignore_index=True
    )
    # print(temp_filename)
    # print("traversal_" + temp_filename)
    res.to_csv(results_dir / temp_filename, index=False)
traversal_mappings.to_csv(results_dir / "traversal_mappings.csv", index=False)"""


def generate_results():
    random.seed(24)
    # traversal_mappings = cudf.DataFrame(columns=["hashable_dict_repr", "filename"])
    traversal_mappings = cudf.DataFrame(
        columns=[
            "UUID",
            "algo",
            "arg1",
            "arg1val",
            "arg2",
            "arg2val",
            "arg3",
            "arg3val",
            "arg4",
            "arg4val",
            "arg5",
            "arg5val",
            "arg6",
            "arg6val",
            "arg7",
            "arg7val",
            "arg8",
            "arg8val",
            "arg9",
            "arg9val",
        ]
    )
    # Generating ALL results files
    for temp in _resultsets:
        res = _resultsets[temp].get_cudf_dataframe()
        # Currently, only traversal results files are generated
        # temp_filename = "traversal-" + str(random.getrandbits(55)) + ".csv"
        temp_filename = str(random.getrandbits(50))
        temp_dict = dict(temp)
        argnames, argvals = [t for t in temp_dict.keys()], [
            t for t in temp_dict.values()
        ]
        single_mapping = np.empty(20, dtype=object)
        dict_length = len(argnames)
        single_mapping[0] = temp_filename
        single_mapping[1] = argvals[0]
        for i in np.arange(1, dict_length):
            single_mapping[2 * i] = argnames[i]
            single_mapping[2 * i + 1] = argvals[i]
        temp_mapping = cudf.DataFrame(
            [single_mapping],
            columns=[
                "UUID",
                "algo",
                "arg1",
                "arg1val",
                "arg2",
                "arg2val",
                "arg3",
                "arg3val",
                "arg4",
                "arg4val",
                "arg5",
                "arg5val",
                "arg6",
                "arg6val",
                "arg7",
                "arg7val",
                "arg8",
                "arg8val",
                "arg9",
                "arg9val",
            ],
        )
        traversal_mappings = cudf.concat(
            [traversal_mappings, temp_mapping], axis=0, ignore_index=True
        )
        res.to_csv(results_dir / ("traversal-" + temp_filename + ".csv"), index=False)
    traversal_mappings.to_csv(
        results_dir / "traversal_mappings.csv", index=False, sep=" "
    )


# generate_results()
