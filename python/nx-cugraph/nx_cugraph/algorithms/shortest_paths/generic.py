# Copyright (c) 2024, NVIDIA CORPORATION.
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
import networkx as nx
import numpy as np

import nx_cugraph as nxcg
from nx_cugraph import _nxver
from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import _dtype_param, _get_float_dtype, networkx_algorithm

from .unweighted import _bfs
from .weighted import _sssp

__all__ = [
    "shortest_path",
    "shortest_path_length",
    "has_path",
]


@networkx_algorithm(version_added="24.04", _plc="bfs")
def has_path(G, source, target):
    # TODO PERF: make faster in core
    try:
        nxcg.bidirectional_shortest_path(G, source, target)
    except nx.NetworkXNoPath:
        return False
    return True


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.04", _plc={"bfs", "sssp"}
)
def shortest_path(
    G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None
):
    """Negative weights are not yet supported."""
    if method not in {"dijkstra", "bellman-ford"}:
        raise ValueError(f"method not supported: {method}")
    if weight is None:
        method = "unweighted"
    if source is None:
        if target is None:
            # All pairs
            if method == "unweighted":
                paths = nxcg.all_pairs_shortest_path(G)
            elif method == "dijkstra":
                paths = nxcg.all_pairs_dijkstra_path(G, weight=weight, dtype=dtype)
            else:  # method == 'bellman-ford':
                paths = nxcg.all_pairs_bellman_ford_path(G, weight=weight, dtype=dtype)
            if _nxver <= (3, 4):
                paths = dict(paths)
        # To target
        elif method == "unweighted":
            paths = nxcg.single_target_shortest_path(G, target)
        else:
            # method == "dijkstra":
            # method == 'bellman-ford':
            # XXX: it seems weird that `reverse_path=True` is necessary here
            G = _to_graph(G, weight, 1, np.float32)
            dtype = _get_float_dtype(dtype, graph=G, weight=weight)
            paths = _sssp(
                G, target, weight, return_type="path", dtype=dtype, reverse_path=True
            )
    elif target is None:
        # From source
        if method == "unweighted":
            paths = nxcg.single_source_shortest_path(G, source)
        elif method == "dijkstra":
            paths = nxcg.single_source_dijkstra_path(
                G, source, weight=weight, dtype=dtype
            )
        else:  # method == 'bellman-ford':
            paths = nxcg.single_source_bellman_ford_path(
                G, source, weight=weight, dtype=dtype
            )
    # From source to target
    elif method == "unweighted":
        paths = nxcg.bidirectional_shortest_path(G, source, target)
    else:
        # method == "dijkstra":
        # method == 'bellman-ford':
        paths = nxcg.bellman_ford_path(G, source, target, weight, dtype=dtype)
    return paths


@shortest_path._can_run
def _(G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(
    extra_params=_dtype_param, version_added="24.04", _plc={"bfs", "sssp"}
)
def shortest_path_length(
    G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None
):
    """Negative weights are not yet supported."""
    if method not in {"dijkstra", "bellman-ford"}:
        raise ValueError(f"method not supported: {method}")
    if weight is None:
        method = "unweighted"
    if source is None:
        if target is None:
            # All pairs
            if method == "unweighted":
                lengths = nxcg.all_pairs_shortest_path_length(G)
            elif method == "dijkstra":
                lengths = nxcg.all_pairs_dijkstra_path_length(
                    G, weight=weight, dtype=dtype
                )
            else:  # method == 'bellman-ford':
                lengths = nxcg.all_pairs_bellman_ford_path_length(
                    G, weight=weight, dtype=dtype
                )
        # To target
        elif method == "unweighted":
            lengths = nxcg.single_target_shortest_path_length(G, target)
            if _nxver <= (3, 4):
                lengths = dict(lengths)
        elif method == "dijkstra":
            lengths = nxcg.single_source_dijkstra_path_length(
                G, target, weight=weight, dtype=dtype
            )
        else:  # method == 'bellman-ford':
            lengths = nxcg.single_source_bellman_ford_path_length(
                G, target, weight=weight, dtype=dtype
            )
    elif target is None:
        # From source
        if method == "unweighted":
            lengths = nxcg.single_source_shortest_path_length(G, source)
        elif method == "dijkstra":
            lengths = nxcg.single_source_dijkstra_path_length(
                G, source, weight=weight, dtype=dtype
            )
        else:  # method == 'bellman-ford':
            lengths = nxcg.single_source_bellman_ford_path_length(
                G, source, weight=weight, dtype=dtype
            )
    # From source to target
    elif method == "unweighted":
        G = _to_graph(G)
        lengths = _bfs(G, source, None, "Source", return_type="length", target=target)
    elif method == "dijkstra":
        lengths = nxcg.dijkstra_path_length(G, source, target, weight, dtype=dtype)
    else:  # method == 'bellman-ford':
        lengths = nxcg.bellman_ford_path_length(G, source, target, weight, dtype=dtype)
    return lengths


@shortest_path_length._can_run
def _(G, source=None, target=None, weight=None, method="dijkstra", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )
