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

import numpy as np
import networkx as nx

import cudf
import cugraph
from cugraph.datasets import dolphins, netscience, karate_disjoint, karate

# from cugraph.testing import utils, Resultset, SMALL_DATASETS, results_dir_path
from cugraph.testing import (
    utils,
    Resultset,
    SMALL_DATASETS,
    default_resultset_download_dir,
)

_resultsets = {}


def add_resultset(result_data_dictionary, **kwargs):
    rs = Resultset(result_data_dictionary)
    hashable_dict_repr = tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
    _resultsets[hashable_dict_repr] = rs


if __name__ == "__main__":
    # =============================================================================
    # Parameters
    # =============================================================================
    SEEDS = [42]

    DIRECTED_GRAPH_OPTIONS = [True, False]

    DEPTH_LIMITS = [None, 1, 5, 18]

    DATASETS = [dolphins, netscience, karate_disjoint]

    # =============================================================================
    # tests/traversal/test_bfs.py
    # =============================================================================
    test_bfs_results = {}

    for ds in DATASETS + [karate]:
        for seed in SEEDS:
            for depth_limit in DEPTH_LIMITS:
                for dirctd in DIRECTED_GRAPH_OPTIONS:
                    # this is used for get_cu_graph_golden_results_and_params
                    Gnx = utils.generate_nx_graph_from_file(
                        ds.get_path(), directed=dirctd
                    )
                    random.seed(seed)
                    start_vertex = random.sample(list(Gnx.nodes()), 1)[0]
                    golden_values = nx.single_source_shortest_path_length(
                        Gnx, start_vertex, cutoff=depth_limit
                    )
                    vertices = cudf.Series(golden_values.keys())
                    distances = cudf.Series(golden_values.values())
                    add_resultset(
                        {"vertex": vertices, "distance": distances},
                        graph_dataset=ds.metadata["name"],
                        graph_directed=str(dirctd),
                        algo="single_source_shortest_path_length",
                        start_vertex=str(start_vertex),
                        cutoff=str(depth_limit),
                    )

    # these are pandas dataframes
    for dirctd in DIRECTED_GRAPH_OPTIONS:
        Gnx = utils.generate_nx_graph_from_file(karate.get_path(), directed=dirctd)
        golden_result = cugraph.bfs_edges(Gnx, source=7)
        cugraph_df = cudf.from_pandas(golden_result)
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

    for ds in SMALL_DATASETS:
        for source in SOURCES:
            Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=True)
            golden_paths = nx.single_source_dijkstra_path_length(Gnx, source)
            vertices = cudf.Series(golden_paths.keys())
            distances = cudf.Series(golden_paths.values())
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
            golden_paths_datatypeconv = nx.single_source_dijkstra_path_length(
                Gnx, source
            )
            vertices_datatypeconv = cudf.Series(golden_paths_datatypeconv.keys())
            distances_datatypeconv = cudf.Series(golden_paths_datatypeconv.values())
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
            add_resultset(
                cugraph.sssp(Gnx, source),
                graph_dataset="karate",
                graph_directed=str(dirctd),
                algo="sssp_nonnative",
                source=str(source),
            )

    Gnx = nx.Graph()
    Gnx.add_edge(0, 1, other=10)
    Gnx.add_edge(1, 2, other=20)
    df = cugraph.sssp(Gnx, 0, edge_attr="other")
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

    # NOTE: Currently, only traversal result files are generated
    random.seed(24)
    traversal_mappings = cudf.DataFrame(
        columns=[
            "#UUID",
            "arg0",
            "arg0val",
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
    results_dir_path = default_resultset_download_dir.path
    if not results_dir_path.exists():
        results_dir_path.mkdir(parents=True, exist_ok=True)

    for temp in _resultsets:
        res = _resultsets[temp].get_cudf_dataframe()
        temp_filename = str(random.getrandbits(50))
        temp_dict = dict(temp)
        argnames, argvals = [t for t in temp_dict.keys()], [
            t for t in temp_dict.values()
        ]
        single_mapping = np.empty(21, dtype=object)
        dict_length = len(argnames)

        single_mapping[0] = temp_filename
        for i in np.arange(dict_length):
            single_mapping[2 * i + 1] = argnames[i]
            single_mapping[2 * i + 2] = argvals[i]
        temp_mapping = cudf.DataFrame(
            [single_mapping],
            columns=[
                "#UUID",
                "arg0",
                "arg0val",
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
        res.to_csv(results_dir_path / (temp_filename + ".csv"), index=False)
    traversal_mappings.to_csv(
        results_dir_path / "traversal_mappings.csv", index=False, sep=" "
    )
