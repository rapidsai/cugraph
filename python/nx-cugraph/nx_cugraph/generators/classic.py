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
import itertools
from numbers import Integral

import cupy as cp
import networkx as nx
import numpy as np

import nx_cugraph as nxcg

from ..utils import _get_int_dtype, index_dtype, networkx_algorithm
from ._utils import (
    _IS_NX32_OR_LESS,
    _common_small_graph,
    _complete_graph_indices,
    _create_using_class,
    _ensure_int,
    _ensure_nonnegative_int,
    _number_and_nodes,
)

__all__ = [
    "barbell_graph",
    "circular_ladder_graph",
    "complete_graph",
    "complete_multipartite_graph",
    "cycle_graph",
    "empty_graph",
    "ladder_graph",
    "lollipop_graph",
    "null_graph",
    "path_graph",
    "star_graph",
    "tadpole_graph",
    "trivial_graph",
    "turan_graph",
    "wheel_graph",
]

concat = itertools.chain.from_iterable


@networkx_algorithm
def barbell_graph(m1, m2, create_using=None):
    # Like two complete graphs and a path_graph
    m1 = _ensure_nonnegative_int(m1)
    if m1 < 2:
        raise nx.NetworkXError("Invalid graph description, m1 should be >=2")
    m2 = _ensure_nonnegative_int(m2)
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_bell1, dst_bell1 = _complete_graph_indices(m1)
    src_bell2 = src_bell1 + (m1 + m2)
    dst_bell2 = dst_bell1 + (m1 + m2)
    if m2 == 0:
        src_bar = cp.array([m1 - 1, m1], index_dtype)
        dst_bar = cp.array([m1, m1 - 1], index_dtype)
    else:
        src_bar = cp.arange(2 * m1 - 1, 2 * m1 + 2 * m2 + 1, dtype=index_dtype) // 2
        dst_bar = (
            cp.arange(m1 - 1, m1 + m2 + 1, dtype=index_dtype)[:, None]
            + cp.array([-1, 1], index_dtype)
        ).ravel()[1:-1]
    src_indices = cp.hstack((src_bell1, src_bar, src_bell2))
    dst_indices = cp.hstack((dst_bell1, dst_bar, dst_bell2))
    G = graph_class.from_coo(2 * m1 + m2, src_indices, dst_indices)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm
def circular_ladder_graph(n, create_using=None):
    return _ladder_graph(n, create_using, is_circular=True)


@networkx_algorithm(nodes_or_number=0)
def complete_graph(n, create_using=None):
    n, nodes = _number_and_nodes(n)
    if n < 3:
        return _common_small_graph(n, nodes, create_using)
    graph_class, inplace = _create_using_class(create_using)
    src_indices, dst_indices = _complete_graph_indices(n)
    G = graph_class.from_coo(n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm
def complete_multipartite_graph(*subset_sizes):
    if not subset_sizes:
        return nxcg.Graph()
    try:
        subset_sizes = [_ensure_int(size) for size in subset_sizes]
    except TypeError:
        subsets = [list(subset) for subset in subset_sizes]
        subset_sizes = [len(subset) for subset in subsets]
        nodes = list(concat(subsets))
    else:
        subsets = nodes = None
        try:
            subset_sizes = [_ensure_nonnegative_int(size) for size in subset_sizes]
        except nx.NetworkXError:
            if _IS_NX32_OR_LESS:
                raise NotImplementedError("Negative number of nodes is not supported")
            raise
    L1 = []
    L2 = []
    total = 0
    for size in subset_sizes:
        all_indices = cp.indices((total, size), dtype=index_dtype)
        L1.append(all_indices[0].ravel())
        L2.append(all_indices[1].ravel() + total)
        total += size
    src_indices = cp.hstack(L1 + L2)
    dst_indices = cp.hstack(L2 + L1)
    subsets_array = cp.array(
        np.repeat(
            np.arange(len(subset_sizes), dtype=_get_int_dtype(len(subset_sizes) - 1)),
            subset_sizes,
        )
    )
    return nxcg.Graph.from_coo(
        subsets_array.size,
        src_indices,
        dst_indices,
        node_values={"subset": subsets_array},
        id_to_key=nodes,
    )


@networkx_algorithm(nodes_or_number=0)
def cycle_graph(n, create_using=None):
    n, nodes = _number_and_nodes(n)
    graph_class, inplace = _create_using_class(create_using)
    if n == 1:
        src_indices = cp.zeros(1, index_dtype)
        dst_indices = cp.zeros(1, index_dtype)
    elif n == 2 and graph_class.is_multigraph() and not graph_class.is_directed():
        # This is kind of a peculiar edge case
        src_indices = cp.array([0, 0, 1, 1], index_dtype)
        dst_indices = cp.array([1, 1, 0, 0], index_dtype)
    elif n < 3:
        return _common_small_graph(n, nodes, create_using)
    elif graph_class.is_directed():
        src_indices = cp.arange(n, dtype=index_dtype)
        dst_indices = cp.arange(1, n + 1, dtype=index_dtype)
        dst_indices[-1] = 0
    else:
        src_indices = cp.arange(2 * n, dtype=index_dtype) // 2
        dst_indices = (
            cp.arange(n, dtype=index_dtype)[:, None] + cp.array([-1, 1], index_dtype)
        ).ravel()
        dst_indices[0] = n - 1
        dst_indices[-1] = 0
    G = graph_class.from_coo(n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(nodes_or_number=0)
def empty_graph(n=0, create_using=None, default=nx.Graph):
    n, nodes = _number_and_nodes(n)
    graph_class, inplace = _create_using_class(create_using, default=default)
    G = graph_class.from_coo(
        n, cp.empty(0, index_dtype), cp.empty(0, index_dtype), id_to_key=nodes
    )
    if inplace:
        return create_using._become(G)
    return G


def _ladder_graph(n, create_using, *, is_circular=False):
    # Like path path_graph with extra arange, and middle link missing
    n = _ensure_nonnegative_int(n)
    if n < 2:
        if not is_circular:
            return _common_small_graph(2 * n, None, create_using, allow_directed=False)
        graph_class, inplace = _create_using_class(create_using)
        if graph_class.is_directed():
            raise nx.NetworkXError("Directed Graph not supported")
        if n == 1:
            src_indices = cp.array([0, 1, 0, 1], index_dtype)
            dst_indices = cp.array([0, 0, 1, 1], index_dtype)
            nodes = None
        elif graph_class.is_multigraph():
            src_indices = cp.array([0, 0, 1, 1], index_dtype)
            dst_indices = cp.array([1, 1, 0, 0], index_dtype)
            nodes = [0, -1]
        else:
            src_indices = cp.array([0, 1], index_dtype)
            dst_indices = cp.array([1, 0], index_dtype)
            nodes = [0, -1]
        G = graph_class.from_coo(2, src_indices, dst_indices, id_to_key=nodes)
        if inplace:
            return create_using._become(G)
        return G
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    path_src = cp.arange(1, 2 * n - 1, dtype=index_dtype) // 2
    path_dst = (
        cp.arange(n, dtype=index_dtype)[:, None] + cp.array([-1, 1], index_dtype)
    ).ravel()[1:-1]
    srcs = [path_src, path_src + n, cp.arange(2 * n, dtype=index_dtype)]
    dsts = [
        path_dst,
        path_dst + n,
        cp.arange(n, 2 * n, dtype=index_dtype),
        cp.arange(0, n, dtype=index_dtype),
    ]
    if is_circular and (n > 2 or graph_class.is_multigraph()):
        srcs.append(cp.array([0, n - 1, n, 2 * n - 1], index_dtype))
        dsts.append(cp.array([n - 1, 0, 2 * n - 1, n], index_dtype))
    src_indices = cp.hstack(srcs)
    dst_indices = cp.hstack(dsts)
    G = graph_class.from_coo(2 * n, src_indices, dst_indices)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm
def ladder_graph(n, create_using=None):
    return _ladder_graph(n, create_using)


@networkx_algorithm(nodes_or_number=[0, 1])
def lollipop_graph(m, n, create_using=None):
    # Like complete_graph then path_graph
    orig_m, unused_nodes_m = m
    orig_n, unused_nodes_n = n
    m, m_nodes = _number_and_nodes(m)
    if m < 2:
        raise nx.NetworkXError(
            "Invalid description: m should indicate at least 2 nodes"
        )
    n, n_nodes = _number_and_nodes(n)
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    msrc_indices, mdst_indices = _complete_graph_indices(m)
    nsrc_indices = cp.arange(2 * m - 1, 2 * m + 2 * n - 1, dtype=index_dtype) // 2
    ndst_indices = (
        cp.arange(m - 1, m + n, dtype=index_dtype)[:, None]
        + cp.array([-1, 1], index_dtype)
    ).ravel()[1:-1]
    src_indices = cp.hstack((msrc_indices, nsrc_indices))
    dst_indices = cp.hstack((mdst_indices, ndst_indices))
    if isinstance(orig_m, Integral) and isinstance(orig_n, Integral):
        nodes = None
    else:
        nodes = list(range(m)) if m_nodes is None else m_nodes
        nodes.extend(range(n) if n_nodes is None else n_nodes)
        if len(set(nodes)) != len(nodes):
            raise nx.NetworkXError("Nodes must be distinct in containers m and n")
    G = graph_class.from_coo(m + n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm
def null_graph(create_using=None):
    return _common_small_graph(0, None, create_using)


@networkx_algorithm(nodes_or_number=0)
def path_graph(n, create_using=None):
    n, nodes = _number_and_nodes(n)
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        src_indices = cp.arange(n - 1, dtype=index_dtype)
        dst_indices = cp.arange(1, n, dtype=index_dtype)
    elif n < 3:
        return _common_small_graph(n, nodes, create_using)
    else:
        src_indices = cp.arange(1, 2 * n - 1, dtype=index_dtype) // 2
        dst_indices = (
            cp.arange(n, dtype=index_dtype)[:, None] + cp.array([-1, 1], index_dtype)
        ).ravel()[1:-1]
    G = graph_class.from_coo(n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(nodes_or_number=0)
def star_graph(n, create_using=None):
    orig_n, orig_nodes = n
    n, nodes = _number_and_nodes(n)
    # star_graph behaves differently whether the input was an int or iterable
    if isinstance(orig_n, Integral):
        if nodes is not None:
            nodes.append(n)
        n += 1
    if n < 3:
        return _common_small_graph(n, nodes, create_using, allow_directed=False)
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    flat = cp.zeros(n - 1, index_dtype)
    ramp = cp.arange(1, n, dtype=index_dtype)
    src_indices = cp.hstack((flat, ramp))
    dst_indices = cp.hstack((ramp, flat))
    G = graph_class.from_coo(n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(nodes_or_number=[0, 1])
def tadpole_graph(m, n, create_using=None):
    orig_m, unused_nodes_m = m
    orig_n, unused_nodes_n = n
    m, m_nodes = _number_and_nodes(m)
    if m < 2:
        raise nx.NetworkXError(
            "Invalid description: m should indicate at least 2 nodes"
        )
    n, n_nodes = _number_and_nodes(n)
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    if isinstance(orig_m, Integral) and isinstance(orig_n, Integral):
        nodes = None
    else:
        nodes = list(range(m)) if m_nodes is None else m_nodes
        nodes.extend(range(n) if n_nodes is None else n_nodes)
    if m == 2 and not graph_class.is_multigraph():
        src_indices = cp.arange(1, 2 * (m + n) - 1, dtype=index_dtype) // 2
        dst_indices = (
            cp.arange((m + n), dtype=index_dtype)[:, None]
            + cp.array([-1, 1], index_dtype)
        ).ravel()[1:-1]
    else:
        src_indices = cp.arange(2 * (m + n), dtype=index_dtype) // 2
        dst_indices = (
            cp.arange((m + n), dtype=index_dtype)[:, None]
            + cp.array([-1, 1], index_dtype)
        ).ravel()
        dst_indices[0] = m - 1
        dst_indices[-1] = 0
    G = graph_class.from_coo(m + n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm
def trivial_graph(create_using=None):
    return _common_small_graph(1, None, create_using)


@networkx_algorithm
def turan_graph(n, r):
    if not 1 <= r <= n:
        raise nx.NetworkXError("Must satisfy 1 <= r <= n")
    n_div_r, n_mod_r = divmod(n, r)
    partitions = [n_div_r] * (r - n_mod_r) + [n_div_r + 1] * n_mod_r
    return complete_multipartite_graph(*partitions)


@networkx_algorithm(nodes_or_number=0)
def wheel_graph(n, create_using=None):
    n, nodes = _number_and_nodes(n)
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    if n < 2:
        G = graph_class.from_coo(
            n, cp.empty(0, index_dtype), cp.empty(0, index_dtype), id_to_key=nodes
        )
    else:
        # Like star_graph
        flat = cp.zeros(n - 1, index_dtype)
        ramp = cp.arange(1, n, dtype=index_dtype)
        # Like cycle_graph
        if n < 3:
            src_indices = cp.empty(0, index_dtype)
            dst_indices = cp.empty(0, index_dtype)
        elif n > 3:
            src_indices = cp.arange(2, 2 * n, dtype=index_dtype) // 2
            dst_indices = (
                cp.arange(1, n, dtype=index_dtype)[:, None]
                + cp.array([-1, 1], index_dtype)
            ).ravel()
            dst_indices[-1] = 1
            dst_indices[0] = n - 1
        elif graph_class.is_multigraph():
            src_indices = cp.array([1, 1, 2, 2], index_dtype)
            dst_indices = cp.array([2, 2, 1, 1], index_dtype)
        else:
            src_indices = cp.array([1, 2], index_dtype)
            dst_indices = cp.array([2, 1], index_dtype)
        src_indices = cp.hstack((flat, ramp, src_indices))
        dst_indices = cp.hstack((ramp, flat, dst_indices))
        G = graph_class.from_coo(n, src_indices, dst_indices, id_to_key=nodes)
    if inplace:
        return create_using._become(G)
    return G
