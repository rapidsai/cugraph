# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
"""Tell NetworkX about the cugraph backend. This file can update itself:

$ make plugin-info

or

$ make all  # Recommended - runs 'plugin-info' followed by 'lint'

or

$ python _nx_cugraph/__init__.py
"""

# Entries between BEGIN and END are automatically generated
_info = {
    "backend_name": "cugraph",
    "project": "nx-cugraph",
    "package": "nx_cugraph",
    "url": "https://github.com/rapidsai/cugraph/tree/branch-24.04/python/nx-cugraph",
    "short_summary": "GPU-accelerated backend.",
    # "description": "TODO",
    "functions": {
        # BEGIN: functions
        "all_pairs_bellman_ford_path",
        "all_pairs_bellman_ford_path_length",
        "all_pairs_shortest_path",
        "all_pairs_shortest_path_length",
        "ancestors",
        "average_clustering",
        "barbell_graph",
        "bellman_ford_path",
        "bellman_ford_path_length",
        "betweenness_centrality",
        "bfs_edges",
        "bfs_layers",
        "bfs_predecessors",
        "bfs_successors",
        "bfs_tree",
        "bidirectional_shortest_path",
        "bull_graph",
        "caveman_graph",
        "chvatal_graph",
        "circular_ladder_graph",
        "clustering",
        "complement",
        "complete_bipartite_graph",
        "complete_graph",
        "complete_multipartite_graph",
        "connected_components",
        "core_number",
        "cubical_graph",
        "cycle_graph",
        "davis_southern_women_graph",
        "degree_centrality",
        "desargues_graph",
        "descendants",
        "descendants_at_distance",
        "diamond_graph",
        "dodecahedral_graph",
        "edge_betweenness_centrality",
        "eigenvector_centrality",
        "empty_graph",
        "florentine_families_graph",
        "from_pandas_edgelist",
        "from_scipy_sparse_array",
        "frucht_graph",
        "generic_bfs_edges",
        "has_path",
        "heawood_graph",
        "hits",
        "house_graph",
        "house_x_graph",
        "icosahedral_graph",
        "in_degree_centrality",
        "is_arborescence",
        "is_branching",
        "is_connected",
        "is_forest",
        "is_isolate",
        "is_tree",
        "is_weakly_connected",
        "isolates",
        "k_truss",
        "karate_club_graph",
        "katz_centrality",
        "krackhardt_kite_graph",
        "ladder_graph",
        "les_miserables_graph",
        "lollipop_graph",
        "louvain_communities",
        "moebius_kantor_graph",
        "node_connected_component",
        "null_graph",
        "number_connected_components",
        "number_of_isolates",
        "number_of_selfloops",
        "number_weakly_connected_components",
        "octahedral_graph",
        "out_degree_centrality",
        "overall_reciprocity",
        "pagerank",
        "pappus_graph",
        "path_graph",
        "petersen_graph",
        "reciprocity",
        "reverse",
        "sedgewick_maze_graph",
        "shortest_path",
        "shortest_path_length",
        "single_source_bellman_ford",
        "single_source_bellman_ford_path",
        "single_source_bellman_ford_path_length",
        "single_source_shortest_path",
        "single_source_shortest_path_length",
        "single_target_shortest_path",
        "single_target_shortest_path_length",
        "star_graph",
        "tadpole_graph",
        "tetrahedral_graph",
        "transitivity",
        "triangles",
        "trivial_graph",
        "truncated_cube_graph",
        "truncated_tetrahedron_graph",
        "turan_graph",
        "tutte_graph",
        "weakly_connected_components",
        "wheel_graph",
        # END: functions
    },
    "additional_docs": {
        # BEGIN: additional_docs
        "all_pairs_bellman_ford_path": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "all_pairs_bellman_ford_path_length": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "average_clustering": "Directed graphs and `weight` parameter are not yet supported.",
        "bellman_ford_path": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "bellman_ford_path_length": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "betweenness_centrality": "`weight` parameter is not yet supported, and RNG with seed may be different.",
        "bfs_edges": "`sort_neighbors` parameter is not yet supported.",
        "bfs_predecessors": "`sort_neighbors` parameter is not yet supported.",
        "bfs_successors": "`sort_neighbors` parameter is not yet supported.",
        "bfs_tree": "`sort_neighbors` parameter is not yet supported.",
        "clustering": "Directed graphs and `weight` parameter are not yet supported.",
        "core_number": "Directed graphs are not yet supported.",
        "edge_betweenness_centrality": "`weight` parameter is not yet supported, and RNG with seed may be different.",
        "eigenvector_centrality": "`nstart` parameter is not used, but it is checked for validity.",
        "from_pandas_edgelist": "cudf.DataFrame inputs also supported; value columns with str is unsuppported.",
        "generic_bfs_edges": "`neighbors` and `sort_neighbors` parameters are not yet supported.",
        "k_truss": (
            "Currently raises `NotImplementedError` for graphs with more than one connected\n"
            "component when k >= 3. We expect to fix this soon."
        ),
        "katz_centrality": "`nstart` isn't used (but is checked), and `normalized=False` is not supported.",
        "louvain_communities": "`seed` parameter is currently ignored, and self-loops are not yet supported.",
        "pagerank": "`dangling` parameter is not supported, but it is checked for validity.",
        "shortest_path": "Negative weights are not yet supported, and method is ununsed.",
        "shortest_path_length": "Negative weights are not yet supported, and method is ununsed.",
        "single_source_bellman_ford": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "single_source_bellman_ford_path": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "single_source_bellman_ford_path_length": "Negative cycles are not yet supported. ``NotImplementedError`` will be raised if there are negative edge weights. We plan to support negative edge weights soon. Also, callable ``weight`` argument is not supported.",
        "transitivity": "Directed graphs are not yet supported.",
        # END: additional_docs
    },
    "additional_parameters": {
        # BEGIN: additional_parameters
        "all_pairs_bellman_ford_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "all_pairs_bellman_ford_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "bellman_ford_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "bellman_ford_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "eigenvector_centrality": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "hits": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
            'weight : string or None, optional (default="weight")': "The edge attribute to use as the edge weight.",
        },
        "katz_centrality": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "louvain_communities": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
            "max_level : int, optional": "Upper limit of the number of macro-iterations (max: 500).",
        },
        "pagerank": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "shortest_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "shortest_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "single_source_bellman_ford": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "single_source_bellman_ford_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "single_source_bellman_ford_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        # END: additional_parameters
    },
}


def get_info():
    """Target of ``networkx.plugin_info`` entry point.

    This tells NetworkX about the cugraph backend without importing nx_cugraph.
    """
    # Convert to e.g. `{"functions": {"myfunc": {"additional_docs": ...}}}`
    d = _info.copy()
    info_keys = {"additional_docs", "additional_parameters"}
    d["functions"] = {
        func: {
            info_key: vals[func]
            for info_key in info_keys
            if func in (vals := d[info_key])
        }
        for func in d["functions"]
    }
    # Add keys for Networkx <3.3
    for func_info in d["functions"].values():
        if "additional_docs" in func_info:
            func_info["extra_docstring"] = func_info["additional_docs"]
        if "additional_parameters" in func_info:
            func_info["extra_parameters"] = func_info["additional_parameters"]

    for key in info_keys:
        del d[key]
    return d


# FIXME: can this use the standard VERSION file and update mechanism?
__version__ = "24.04.00"

if __name__ == "__main__":
    from pathlib import Path

    from _nx_cugraph.core import main

    filepath = Path(__file__)
    text = main(filepath)
    with filepath.open("w") as f:
        f.write(text)
