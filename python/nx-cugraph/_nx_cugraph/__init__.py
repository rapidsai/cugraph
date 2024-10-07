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
import os

from _nx_cugraph._version import __version__

# This is normally handled by packaging.version.Version, but instead of adding
# an additional runtime dependency on "packaging", assume __version__ will
# always be in <major>.<minor>.<build> format.
(_version_major, _version_minor) = __version__.split(".")[:2]

# Entries between BEGIN and END are automatically generated
_info = {
    "backend_name": "cugraph",
    "project": "nx-cugraph",
    "package": "nx_cugraph",
    "url": f"https://rapids.ai/nx-cugraph",
    "short_summary": "GPU-accelerated backend.",
    # "description": "TODO",
    "functions": {
        # BEGIN: functions
        "all_pairs_bellman_ford_path",
        "all_pairs_bellman_ford_path_length",
        "all_pairs_dijkstra",
        "all_pairs_dijkstra_path",
        "all_pairs_dijkstra_path_length",
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
        "convert_node_labels_to_integers",
        "core_number",
        "cubical_graph",
        "cycle_graph",
        "davis_southern_women_graph",
        "degree_centrality",
        "desargues_graph",
        "descendants",
        "descendants_at_distance",
        "diamond_graph",
        "dijkstra_path",
        "dijkstra_path_length",
        "dodecahedral_graph",
        "edge_betweenness_centrality",
        "ego_graph",
        "eigenvector_centrality",
        "empty_graph",
        "florentine_families_graph",
        "from_dict_of_lists",
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
        "is_negatively_weighted",
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
        "relabel_nodes",
        "reverse",
        "sedgewick_maze_graph",
        "shortest_path",
        "shortest_path_length",
        "single_source_bellman_ford",
        "single_source_bellman_ford_path",
        "single_source_bellman_ford_path_length",
        "single_source_dijkstra",
        "single_source_dijkstra_path",
        "single_source_dijkstra_path_length",
        "single_source_shortest_path",
        "single_source_shortest_path_length",
        "single_target_shortest_path",
        "single_target_shortest_path_length",
        "star_graph",
        "tadpole_graph",
        "tetrahedral_graph",
        "to_dict_of_lists",
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
        "ego_graph": "Weighted ego_graph with negative cycles is not yet supported. `NotImplementedError` will be raised if there are negative `distance` edge weights.",
        "eigenvector_centrality": "`nstart` parameter is not used, but it is checked for validity.",
        "from_pandas_edgelist": "cudf.DataFrame inputs also supported; value columns with str is unsuppported.",
        "generic_bfs_edges": "`neighbors` and `sort_neighbors` parameters are not yet supported.",
        "katz_centrality": "`nstart` isn't used (but is checked), and `normalized=False` is not supported.",
        "louvain_communities": "`seed` parameter is currently ignored, and self-loops are not yet supported.",
        "pagerank": "`dangling` parameter is not supported, but it is checked for validity.",
        "shortest_path": "Negative weights are not yet supported.",
        "shortest_path_length": "Negative weights are not yet supported.",
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
        "all_pairs_dijkstra": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "all_pairs_dijkstra_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "all_pairs_dijkstra_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "bellman_ford_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "bellman_ford_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "dijkstra_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "dijkstra_path_length": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "ego_graph": {
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
        "single_source_dijkstra": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "single_source_dijkstra_path": {
            "dtype : dtype or None, optional": "The data type (np.float32, np.float64, or None) to use for the edge weights in the algorithm. If None, then dtype is determined by the edge values.",
        },
        "single_source_dijkstra_path_length": {
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

    d["default_config"] = {
        "use_compat_graphs": os.environ.get("NX_CUGRAPH_USE_COMPAT_GRAPHS", "true")
        .strip()
        .lower()
        == "true",
    }

    # Enable zero-code change usage with a simple environment variable
    # by setting or updating other NETWORKX environment variables.
    if os.environ.get("NX_CUGRAPH_AUTOCONFIG", "").strip().lower() == "true":
        from itertools import chain

        def update_env_var(varname):
            """Add "cugraph" to a list of backend names environment variable."""
            if varname not in os.environ:
                os.environ[varname] = "cugraph"
                return
            string = os.environ[varname]
            vals = [
                stripped for x in string.strip().split(",") if (stripped := x.strip())
            ]
            if "cugraph" not in vals:
                # Should we append or prepend? Let's be first!
                os.environ[varname] = ",".join(chain(["cugraph"], vals))

        # Automatically convert NetworkX Graphs to nx-cugraph for algorithms
        if (varname := "NETWORKX_BACKEND_PRIORITY_ALGOS") in os.environ:
            # "*_ALGOS" is given priority in NetworkX >=3.4
            update_env_var(varname)
            # But update this too to "just work" if users mix env vars and nx versions
            os.environ["NETWORKX_BACKEND_PRIORITY"] = os.environ[varname]
        else:
            update_env_var("NETWORKX_BACKEND_PRIORITY")
        # And for older NetworkX versions
        update_env_var("NETWORKX_AUTOMATIC_BACKENDS")  # For NetworkX 3.2
        update_env_var("NETWORKX_GRAPH_CONVERT")  # For NetworkX 3.0 and 3.1
        # Automatically create nx-cugraph Graph from graph generators
        update_env_var("NETWORKX_BACKEND_PRIORITY_GENERATORS")
        # Run default NetworkX implementation (in >=3.4) if not implemented by nx-cugraph
        if (varname := "NETWORKX_FALLBACK_TO_NX") not in os.environ:
            os.environ[varname] = "true"
        # Cache graph conversions (default is False in NetworkX 3.2
        if (varname := "NETWORKX_CACHE_CONVERTED_GRAPHS") not in os.environ:
            os.environ[varname] = "true"

    return d


def _check_networkx_version() -> tuple[int, int]:
    """Check the version of networkx and return ``(major, minor)`` version tuple."""
    import re
    import warnings

    import networkx as nx

    version_major, version_minor = nx.__version__.split(".")[:2]
    if version_major != "3":
        warnings.warn(
            f"nx-cugraph version {__version__} is only known to work with networkx "
            f"versions 3.x, but networkx {nx.__version__} is installed. "
            "Perhaps try upgrading your Python environment.",
            UserWarning,
            stacklevel=2,
        )

    # Allow single-digit minor versions, e.g. 3.4 and release candidates, e.g. 3.4rc0
    pattern = r"^\d(rc\d+)?$"

    if not re.match(pattern, version_minor):
        raise RuntimeWarning(
            f"nx-cugraph version {__version__} does not work with networkx version "
            f"{nx.__version__}. Please upgrade (or fix) your Python environment."
        )

    nxver_major = int(version_major)
    nxver_minor = int(re.match(r"^\d+", version_minor).group())
    return (nxver_major, nxver_minor)


if __name__ == "__main__":
    from pathlib import Path

    # This script imports nx_cugraph modules, which imports nx_cugraph runtime
    # dependencies. The modules do not need the runtime deps, so stub them out
    # to avoid installing them.
    class Stub:
        def __getattr__(self, *args, **kwargs):
            return Stub()

        def __call__(self, *args, **kwargs):
            return Stub()

    import sys

    sys.modules["cupy"] = Stub()
    sys.modules["numpy"] = Stub()
    sys.modules["pylibcugraph"] = Stub()

    from _nx_cugraph.core import main

    filepath = Path(__file__)
    text = main(filepath)
    with filepath.open("w") as f:
        f.write(text)
