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

$ make plugin-info  # Recommended method for development

or

$ python _nx_cugraph/__init__.py
"""

# Entries between BEGIN and END are automatically generated
_info = {
    "backend_name": "cugraph",
    "project": "nx-cugraph",
    "package": "nx_cugraph",
    "url": "https://github.com/rapidsai/cugraph/tree/branch-24.02/python/nx-cugraph",
    "short_summary": "GPU-accelerated backend.",
    # "description": "TODO",
    "functions": {
        # BEGIN: functions
        "ancestors",
        "barbell_graph",
        "betweenness_centrality",
        "bfs_edges",
        "bfs_layers",
        "bfs_predecessors",
        "bfs_successors",
        "bfs_tree",
        "bull_graph",
        "caveman_graph",
        "chvatal_graph",
        "circular_ladder_graph",
        "complete_bipartite_graph",
        "complete_graph",
        "complete_multipartite_graph",
        "connected_components",
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
        "heawood_graph",
        "hits",
        "house_graph",
        "house_x_graph",
        "icosahedral_graph",
        "in_degree_centrality",
        "is_connected",
        "is_isolate",
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
        "octahedral_graph",
        "out_degree_centrality",
        "pagerank",
        "pappus_graph",
        "path_graph",
        "petersen_graph",
        "sedgewick_maze_graph",
        "single_source_shortest_path_length",
        "single_target_shortest_path_length",
        "star_graph",
        "tadpole_graph",
        "tetrahedral_graph",
        "trivial_graph",
        "truncated_cube_graph",
        "truncated_tetrahedron_graph",
        "turan_graph",
        "tutte_graph",
        "wheel_graph",
        # END: functions
    },
    "extra_docstrings": {
        # BEGIN: extra_docstrings
        "betweenness_centrality": "`weight` parameter is not yet supported, and RNG with seed may be different.",
        "bfs_edges": "`sort_neighbors` parameter is not yet supported.",
        "bfs_predecessors": "`sort_neighbors` parameter is not yet supported.",
        "bfs_successors": "`sort_neighbors` parameter is not yet supported.",
        "bfs_tree": "`sort_neighbors` parameter is not yet supported.",
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
        # END: extra_docstrings
    },
    "extra_parameters": {
        # BEGIN: extra_parameters
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
        # END: extra_parameters
    },
}


def get_info():
    """Target of ``networkx.plugin_info`` entry point.

    This tells NetworkX about the cugraph backend without importing nx_cugraph.
    """
    # Convert to e.g. `{"functions": {"myfunc": {"extra_docstring": ...}}}`
    d = _info.copy()
    info_keys = {
        "extra_docstrings": "extra_docstring",
        "extra_parameters": "extra_parameters",
    }
    d["functions"] = {
        func: {
            new_key: vals[func]
            for old_key, new_key in info_keys.items()
            if func in (vals := d[old_key])
        }
        for func in d["functions"]
    }
    for key in info_keys:
        del d[key]
    return d


# FIXME: can this use the standard VERSION file and update mechanism?
__version__ = "24.02.00"

if __name__ == "__main__":
    from pathlib import Path

    from _nx_cugraph.core import main

    filepath = Path(__file__)
    text = main(filepath)
    with filepath.open("w") as f:
        f.write(text)
