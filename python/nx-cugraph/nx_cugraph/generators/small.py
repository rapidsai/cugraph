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
import cupy as cp
import networkx as nx

import nx_cugraph as nxcg

from ..utils import index_dtype, networkx_algorithm
from ._utils import _IS_NX32_OR_LESS, _create_using_class

__all__ = [
    "bull_graph",
    "chvatal_graph",
    "cubical_graph",
    "desargues_graph",
    "diamond_graph",
    "dodecahedral_graph",
    "frucht_graph",
    "heawood_graph",
    "house_graph",
    "house_x_graph",
    "icosahedral_graph",
    "krackhardt_kite_graph",
    "moebius_kantor_graph",
    "octahedral_graph",
    "pappus_graph",
    "petersen_graph",
    "sedgewick_maze_graph",
    "tetrahedral_graph",
    "truncated_cube_graph",
    "truncated_tetrahedron_graph",
    "tutte_graph",
]


@networkx_algorithm(version_added="23.12")
def bull_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_indices = cp.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 4], index_dtype)
    dst_indices = cp.array([1, 2, 0, 2, 3, 0, 1, 4, 1, 2], index_dtype)
    G = graph_class.from_coo(5, src_indices, dst_indices, name="Bull Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def chvatal_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
            6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11,
            11, 11,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 4, 6, 9, 0, 2, 5, 7, 1, 3, 6, 8, 2, 4, 7, 9, 0, 3, 5, 8, 1, 4, 10, 11,
            0, 2, 10, 11, 1, 3, 8, 11, 2, 4, 7, 10, 0, 3, 10, 11, 5, 6, 8, 9, 5, 6,
            7, 9,
        ],
        index_dtype,
    )
    # fmt: on
    G = graph_class.from_coo(12, src_indices, dst_indices, name="Chvatal Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def cubical_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_indices = cp.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
        index_dtype,
    )
    dst_indices = cp.array(
        [1, 3, 4, 0, 2, 7, 1, 3, 6, 0, 2, 5, 0, 5, 7, 3, 4, 6, 2, 5, 7, 1, 4, 6],
        index_dtype,
    )
    name = ("Platonic Cubical Graph",) if _IS_NX32_OR_LESS else "Platonic Cubical Graph"
    G = graph_class.from_coo(8, src_indices, dst_indices, name=name)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def desargues_graph(create_using=None):
    # This can also be defined w.r.t. LCF_graph
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
            14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 5, 19, 0, 2, 16, 1, 3, 11, 2, 4, 14, 3, 5, 9, 0, 4, 6, 5, 7, 15, 6, 8,
            18, 7, 9, 13, 4, 8, 10, 9, 11, 19, 2, 10, 12, 11, 13, 17, 8, 12, 14, 3,
            13, 15, 6, 14, 16, 1, 15, 17, 12, 16, 18, 7, 17, 19, 0, 10, 18,
        ],
        index_dtype,
    )
    # fmt: on
    if graph_class.is_multigraph():
        src_indices_extra = cp.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            index_dtype,
        )
        dst_indices_extra = cp.array(
            [5, 16, 11, 14, 9, 0, 15, 18, 13, 4, 19, 2, 17, 8, 3, 6, 1, 12, 7, 10],
            index_dtype,
        )
        src_indices = cp.hstack((src_indices, src_indices_extra))
        dst_indices = cp.hstack((dst_indices, dst_indices_extra))
    G = graph_class.from_coo(20, src_indices, dst_indices, name="Desargues Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def diamond_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_indices = cp.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3], index_dtype)
    dst_indices = cp.array([1, 2, 0, 2, 3, 0, 1, 3, 1, 2], index_dtype)
    G = graph_class.from_coo(4, src_indices, dst_indices, name="Diamond Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def dodecahedral_graph(create_using=None):
    # This can also be defined w.r.t. LCF_graph
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
            14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 10, 19, 0, 2, 8, 1, 3, 6, 2, 4, 19, 3, 5, 17, 4, 6, 15, 2, 5, 7, 6, 8,
            14, 1, 7, 9, 8, 10, 13, 0, 9, 11, 10, 12, 18, 11, 13, 16, 9, 12, 14, 7,
            13, 15, 5, 14, 16, 12, 15, 17, 4, 16, 18, 11, 17, 19, 0, 3, 18,
        ],
        index_dtype,
    )
    # fmt: on
    if graph_class.is_multigraph():
        src_indices_extra = cp.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            index_dtype,
        )
        dst_indices_extra = cp.array(
            [10, 8, 6, 19, 17, 15, 2, 14, 1, 13, 0, 18, 16, 9, 7, 5, 12, 4, 11, 3],
            index_dtype,
        )
        src_indices = cp.hstack((src_indices, src_indices_extra))
        dst_indices = cp.hstack((dst_indices, dst_indices_extra))
    G = graph_class.from_coo(20, src_indices, dst_indices, name="Dodecahedral Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def frucht_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        src_indices = cp.array(
            [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 10],
            index_dtype,
        )
        dst_indices = cp.array(
            [1, 7, 2, 7, 3, 8, 4, 9, 5, 9, 6, 10, 0, 10, 11, 9, 11, 11],
            index_dtype,
        )
    else:
        # fmt: off
        src_indices = cp.array(
            [
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7,
                7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11,
            ],
            index_dtype,
        )
        dst_indices = cp.array(
            [
                1, 6, 7, 0, 2, 7, 1, 3, 8, 2, 4, 9, 3, 5, 9, 4, 6, 10, 0, 5, 10, 0,
                1, 11, 2, 9, 11, 3, 4, 8, 5, 6, 11, 7, 8, 10,
            ],
            index_dtype,
        )
        # fmt: on
    G = graph_class.from_coo(12, src_indices, dst_indices, name="Frucht Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def heawood_graph(create_using=None):
    # This can also be defined w.r.t. LCF_graph
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 5, 13, 0, 2, 10, 1, 3, 7, 2, 4, 12, 3, 5, 9, 0, 4, 6, 5, 7, 11, 2, 6,
            8, 7, 9, 13, 4, 8, 10, 1, 9, 11, 6, 10, 12, 3, 11, 13, 0, 8, 12,
        ],
        index_dtype,
    )
    # fmt: on
    if graph_class.is_multigraph():
        src_indices_extra = cp.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            index_dtype,
        )
        dst_indices_extra = cp.array(
            [5, 10, 7, 12, 9, 0, 11, 2, 13, 4, 1, 6, 3, 8],
            index_dtype,
        )
        src_indices = cp.hstack((src_indices, src_indices_extra))
        dst_indices = cp.hstack((dst_indices, dst_indices_extra))
    G = graph_class.from_coo(14, src_indices, dst_indices, name="Heawood Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def house_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_indices = cp.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4], index_dtype)
    dst_indices = cp.array([1, 2, 0, 3, 0, 3, 4, 1, 2, 4, 2, 3], index_dtype)
    G = graph_class.from_coo(5, src_indices, dst_indices, name="House Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def house_x_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_indices = cp.array(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4], index_dtype
    )
    dst_indices = cp.array(
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 4, 0, 1, 2, 4, 2, 3], index_dtype
    )
    G = graph_class.from_coo(
        5, src_indices, dst_indices, name="House-with-X-inside Graph"
    )
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def icosahedral_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
            4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9,
            9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 5, 7, 8, 11, 0, 2, 5, 6, 8, 1, 3, 6, 8, 9, 2, 4, 6, 9, 10, 3, 5, 6,
            10, 11, 0, 1, 4, 6, 11, 1, 2, 3, 4, 5, 0, 8, 9, 10, 11, 0, 1, 2, 7, 9, 2,
            3, 7, 8, 10, 3, 4, 7, 9, 11, 0, 4, 5, 7, 10,
        ],
        index_dtype,
    )
    # fmt: on
    G = graph_class.from_coo(
        12, src_indices, dst_indices, name="Platonic Icosahedral Graph"
    )
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def krackhardt_kite_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5,
            5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 2, 3, 5, 0, 3, 4, 6, 0, 3, 5, 0, 1, 2, 4, 5, 6, 1, 3, 6, 0, 2, 3, 6,
            7, 1, 3, 4, 5, 7, 5, 6, 8, 7, 9, 8,
        ],
        index_dtype,
    )
    # fmt: on
    G = graph_class.from_coo(
        10, src_indices, dst_indices, name="Krackhardt Kite Social Network"
    )
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def moebius_kantor_graph(create_using=None):
    # This can also be defined w.r.t. LCF_graph
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14,
            14, 14, 15, 15, 15,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 5, 15, 0, 2, 12, 1, 3, 7, 2, 4, 14, 3, 5, 9, 0, 4, 6, 5, 7, 11, 2, 6,
            8, 7, 9, 13, 4, 8, 10, 9, 11, 15, 6, 10, 12, 1, 11, 13, 8, 12, 14, 3, 13,
            15, 0, 10, 14,
        ],
        index_dtype,
    )
    # fmt: on
    if graph_class.is_multigraph():
        src_indices_extra = cp.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            index_dtype,
        )
        dst_indices_extra = cp.array(
            [5, 12, 7, 14, 9, 0, 11, 2, 13, 4, 15, 6, 1, 8, 3, 10],
            index_dtype,
        )
        src_indices = cp.hstack((src_indices, src_indices_extra))
        dst_indices = cp.hstack((dst_indices, dst_indices_extra))
    G = graph_class.from_coo(16, src_indices, dst_indices, name="Moebius-Kantor Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def octahedral_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    src_indices = cp.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        index_dtype,
    )
    dst_indices = cp.array(
        [1, 2, 3, 4, 0, 2, 3, 5, 0, 1, 4, 5, 0, 1, 4, 5, 0, 2, 3, 5, 1, 2, 3, 4],
        index_dtype,
    )
    G = graph_class.from_coo(
        6, src_indices, dst_indices, name="Platonic Octahedral Graph"
    )
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def pappus_graph():
    # This can also be defined w.r.t. LCF_graph
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
            14, 15, 15, 15, 16, 16, 16, 17, 17, 17,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 5, 17, 0, 2, 8, 1, 3, 13, 2, 4, 10, 3, 5, 15, 0, 4, 6, 5, 7, 11, 6, 8,
            14, 1, 7, 9, 8, 10, 16, 3, 9, 11, 6, 10, 12, 11, 13, 17, 2, 12, 14, 7,
            13, 15, 4, 14, 16, 9, 15, 17, 0, 12, 16,
        ],
        index_dtype,
    )
    # fmt: on
    return nxcg.Graph.from_coo(18, src_indices, dst_indices, name="Pappus Graph")


@networkx_algorithm(version_added="23.12")
def petersen_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 4, 5, 0, 2, 6, 1, 3, 7, 2, 4, 8, 0, 3, 9, 0, 7, 8, 1, 8, 9, 2, 5, 9,
            3, 5, 6, 4, 6, 7,
        ],
        index_dtype,
    )
    # fmt: on
    G = graph_class.from_coo(10, src_indices, dst_indices, name="Petersen Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def sedgewick_maze_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        src_indices = cp.array([0, 0, 0, 1, 2, 3, 3, 4, 4, 4], index_dtype)
        dst_indices = cp.array([2, 5, 7, 7, 6, 4, 5, 5, 6, 7], index_dtype)
    else:
        src_indices = cp.array(
            [0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7],
            index_dtype,
        )
        dst_indices = cp.array(
            [2, 5, 7, 7, 0, 6, 4, 5, 3, 5, 6, 7, 0, 3, 4, 2, 4, 0, 1, 4],
            index_dtype,
        )
    G = graph_class.from_coo(8, src_indices, dst_indices, name="Sedgewick Maze")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def tetrahedral_graph(create_using=None):
    # This can also be defined w.r.t. complete_graph
    graph_class, inplace = _create_using_class(create_using)
    src_indices = cp.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], index_dtype)
    dst_indices = cp.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], index_dtype)
    name = (
        "Platonic Tetrahedral graph"
        if _IS_NX32_OR_LESS
        else "Platonic Tetrahedral Graph"
    )
    G = graph_class.from_coo(4, src_indices, dst_indices, name=name)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def truncated_cube_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
            14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20,
            20, 21, 21, 21, 22, 22, 22, 23, 23, 23,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 2, 4, 0, 11, 14, 0, 3, 4, 2, 6, 8, 0, 2, 5, 4, 16, 18, 3, 7, 8, 6, 10,
            12, 3, 6, 9, 8, 17, 20, 7, 11, 12, 1, 10, 14, 7, 10, 13, 12, 21, 22, 1,
            11, 15, 14, 19, 23, 5, 17, 18, 9, 16, 20, 5, 16, 19, 15, 18, 23, 9, 17,
            21, 13, 20, 22, 13, 21, 23, 15, 19, 22,
        ],
        index_dtype,
    )
    # fmt: on
    G = graph_class.from_coo(24, src_indices, dst_indices, name="Truncated Cube Graph")
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def truncated_tetrahedron_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        src_indices = cp.array(
            [0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9, 10], index_dtype
        )
        dst_indices = cp.array(
            [1, 2, 9, 2, 6, 3, 4, 11, 5, 11, 6, 7, 7, 8, 9, 10, 10, 11], index_dtype
        )
    else:
        # fmt: off
        src_indices = cp.array(
            [
                0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7,
                7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11,
            ],
            index_dtype,
        )
        dst_indices = cp.array(
            [
                1, 2, 9, 0, 2, 6, 0, 1, 3, 2, 4, 11, 3, 5, 11, 4, 6, 7, 1, 5, 7, 5,
                6, 8, 7, 9, 10, 0, 8, 10, 8, 9, 11, 3, 4, 10,
            ],
            index_dtype,
        )
        # fmt: on
    G = graph_class.from_coo(
        12, src_indices, dst_indices, name="Truncated Tetrahedron Graph"
    )
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="23.12")
def tutte_graph(create_using=None):
    graph_class, inplace = _create_using_class(create_using)
    if graph_class.is_directed():
        raise nx.NetworkXError("Directed Graph not supported")
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7,
            8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14,
            14, 15, 15, 15, 16, 16, 16, 17, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20,
            20, 21, 21, 21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26,
            26, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30, 31, 31, 31, 32, 32,
            32, 33, 33, 33, 34, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38,
            38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 43, 43, 43, 44, 44,
            44, 45, 45, 45,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 2, 3, 0, 4, 26, 0, 10, 11, 0, 18, 19, 1, 5, 33, 4, 6, 29, 5, 7, 27, 6,
            8, 14, 7, 9, 38, 8, 10, 37, 2, 9, 39, 2, 12, 39, 11, 13, 35, 12, 14, 15,
            7, 13, 34, 13, 16, 22, 15, 17, 44, 16, 18, 43, 3, 17, 45, 3, 20, 45, 19,
            21, 41, 20, 22, 23, 15, 21, 40, 21, 24, 27, 23, 25, 32, 24, 26, 31, 1,
            25, 33, 6, 23, 28, 27, 29, 32, 5, 28, 30, 29, 31, 33, 25, 30, 32, 24, 28,
            31, 4, 26, 30, 14, 35, 38, 12, 34, 36, 35, 37, 39, 9, 36, 38, 8, 34, 37,
            10, 11, 36, 22, 41, 44, 20, 40, 42, 41, 43, 45, 17, 42, 44, 16, 40, 43,
            18, 19, 42,
        ],
        index_dtype,
    )
    # fmt: on
    G = graph_class.from_coo(46, src_indices, dst_indices, name="Tutte's Graph")
    if inplace:
        return create_using._become(G)
    return G
