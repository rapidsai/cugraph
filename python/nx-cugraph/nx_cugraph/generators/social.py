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
import numpy as np

import nx_cugraph as nxcg
from nx_cugraph import _nxver

from ..utils import index_dtype, networkx_algorithm

__all__ = [
    "davis_southern_women_graph",
    "florentine_families_graph",
    "karate_club_graph",
    "les_miserables_graph",
]


@networkx_algorithm(version_added="23.12")
def davis_southern_women_graph():
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3,
            3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8,
            8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
            12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15,
            16, 16, 17, 17, 18, 18, 18, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21,
            21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25,
            25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
            27, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 30, 30, 30,
            31, 31, 31,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            18, 19, 20, 21, 22, 23, 25, 26, 18, 19, 20, 22, 23, 24, 25, 19, 20, 21,
            22, 23, 24, 25, 26, 18, 20, 21, 22, 23, 24, 25, 20, 21, 22, 24, 20, 22,
            23, 25, 22, 23, 24, 25, 23, 25, 26, 22, 24, 25, 26, 24, 25, 26, 29, 25,
            26, 27, 29, 25, 26, 27, 29, 30, 31, 24, 25, 26, 27, 29, 30, 31, 23, 24,
            26, 27, 28, 29, 30, 31, 24, 25, 27, 28, 29, 25, 26, 26, 28, 26, 28, 0, 1,
            3, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 8, 0, 1,
            2, 3, 5, 6, 7, 13, 1, 2, 3, 4, 6, 8, 9, 12, 13, 14, 0, 1, 2, 3, 5, 6, 7,
            8, 9, 10, 11, 12, 14, 15, 0, 2, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 10,
            11, 12, 13, 14, 13, 14, 16, 17, 9, 10, 11, 12, 13, 14, 11, 12, 13, 11,
            12, 13,
        ],
        index_dtype,
    )
    bipartite = cp.array(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1,
        ],
        np.int8,
    )
    women = [
        "Evelyn Jefferson", "Laura Mandeville", "Theresa Anderson", "Brenda Rogers",
        "Charlotte McDowd", "Frances Anderson", "Eleanor Nye", "Pearl Oglethorpe",
        "Ruth DeSand", "Verne Sanderson", "Myra Liddel", "Katherina Rogers",
        "Sylvia Avondale", "Nora Fayette", "Helen Lloyd", "Dorothy Murchison",
        "Olivia Carleton", "Flora Price",
    ]
    events = [
        "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12",
        "E13", "E14",
    ]
    # fmt: on
    use_compat_graph = _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs
    return nxcg.CudaGraph.from_coo(
        32,
        src_indices,
        dst_indices,
        node_values={"bipartite": bipartite},
        id_to_key=women + events,
        top=women,
        bottom=events,
        use_compat_graph=use_compat_graph,
    )


@networkx_algorithm(version_added="23.12")
def florentine_families_graph():
    # fmt: off
    src_indices = cp.array(
        [
            0, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8,
            9, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 13, 14, 14, 14,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            8, 5, 6, 8, 4, 8, 6, 10, 13, 2, 10, 13, 1, 1, 3, 7, 14, 6, 0, 1, 2, 11,
            12, 14, 12, 3, 4, 13, 8, 13, 14, 8, 9, 3, 4, 10, 11, 6, 8, 11,
        ],
        index_dtype,
    )
    nodes = [
        "Acciaiuoli", "Albizzi", "Barbadori", "Bischeri", "Castellani", "Ginori",
        "Guadagni", "Lamberteschi", "Medici", "Pazzi", "Peruzzi", "Ridolfi",
        "Salviati", "Strozzi", "Tornabuoni"
    ]
    # fmt: on
    use_compat_graph = _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs
    return nxcg.CudaGraph.from_coo(
        15,
        src_indices,
        dst_indices,
        id_to_key=nodes,
        use_compat_graph=use_compat_graph,
    )


@networkx_algorithm(version_added="23.12")
def karate_club_graph():
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5,
            6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10, 10, 10, 11, 12, 12, 13,
            13, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 19, 20,
            20, 21, 21, 22, 22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26,
            27, 27, 27, 27, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31,
            31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33,
            33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33, 33,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31, 0, 2, 3, 7, 13,
            17, 19, 21, 30, 0, 1, 3, 7, 8, 9, 13, 27, 28, 32, 0, 1, 2, 7, 12, 13, 0,
            6, 10, 0, 6, 10, 16, 0, 4, 5, 16, 0, 1, 2, 3, 0, 2, 30, 32, 33, 2, 33,
            0, 4, 5, 0, 0, 3, 0, 1, 2, 3, 33, 32, 33, 32, 33, 5, 6, 0, 1, 32, 33, 0,
            1, 33, 32, 33, 0, 1, 32, 33, 25, 27, 29, 32, 33, 25, 27, 31, 23, 24, 31,
            29, 33, 2, 23, 24, 33, 2, 31, 33, 23, 26, 32, 33, 1, 8, 32, 33, 0, 24,
            25, 28, 32, 33, 2, 8, 14, 15, 18, 20, 22, 23, 29, 30, 31, 33, 8, 9, 13,
            14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30, 31, 32,
        ],
        index_dtype,
    )
    weights = cp.array(
        [
            4, 5, 3, 3, 3, 3, 2, 2, 2, 3, 1, 3, 2, 2, 2, 2, 4, 6, 3, 4, 5, 1, 2, 2,
            2, 5, 6, 3, 4, 5, 1, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 5, 3, 3,
            3, 2, 5, 3, 2, 4, 4, 3, 2, 5, 3, 3, 4, 1, 2, 2, 3, 3, 3, 1, 3, 3, 5, 3,
            3, 3, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 2, 2, 1, 3, 1, 2, 2, 2, 3, 5, 4, 3,
            5, 4, 2, 3, 2, 5, 2, 7, 4, 2, 2, 4, 3, 4, 2, 2, 2, 3, 4, 4, 2, 2, 3, 3,
            3, 2, 2, 7, 2, 4, 4, 2, 3, 3, 3, 1, 3, 2, 5, 4, 3, 4, 5, 4, 2, 3, 2, 4,
            2, 1, 1, 3, 4, 2, 4, 2, 2, 3, 4, 5,
        ],
        np.int8,
    )
    # For now, cupy doesn't handle str dtypes and we primarily handle cupy arrays.
    # We try to support numpy arrays for node values, so let's use numpy here.
    clubs = np.array([
        "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi",
        "Mr. Hi", "Mr. Hi", "Officer", "Mr. Hi", "Mr. Hi", "Mr. Hi", "Mr. Hi",
        "Officer", "Officer", "Mr. Hi", "Mr. Hi", "Officer", "Mr. Hi", "Officer",
        "Mr. Hi", "Officer", "Officer", "Officer", "Officer", "Officer", "Officer",
        "Officer", "Officer", "Officer", "Officer", "Officer", "Officer",
    ])
    # fmt: on
    use_compat_graph = _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs
    return nxcg.CudaGraph.from_coo(
        34,
        src_indices,
        dst_indices,
        edge_values={"weight": weights},
        node_values={"club": clubs},
        name="Zachary's Karate Club",
        use_compat_graph=use_compat_graph,
    )


@networkx_algorithm(version_added="23.12")
def les_miserables_graph():
    # fmt: off
    src_indices = cp.array(
        [
            0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10,
            10, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13, 14, 14, 15, 15, 15,
            15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17,
            17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19,
            20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 23, 23, 23,
            23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26,
            26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28,
            28, 28, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 30,
            30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31,
            31, 31, 31, 31, 31, 31, 32, 33, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35,
            35, 35, 35, 35, 35, 35, 35, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
            38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39,
            40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 42, 42, 42, 42, 42,
            42, 43, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 46,
            46, 46, 46, 46, 46, 46, 47, 47, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49,
            49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 51, 51, 51, 51,
            51, 51, 51, 52, 53, 53, 54, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 57,
            57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59,
            59, 59, 59, 60, 60, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 64,
            65, 65, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 69, 69, 69,
            69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71,
            71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73,
            73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
            73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 75, 75, 75, 76, 76,
            76, 76, 76, 76, 76,
        ],
        index_dtype,
    )
    dst_indices = cp.array(
        [
            25, 58, 70, 9, 15, 25, 31, 37, 39, 58, 59, 70, 73, 6, 17, 21, 24, 30, 31,
            35, 40, 46, 49, 55, 67, 8, 10, 12, 16, 27, 39, 42, 73, 34, 49, 23, 26,
            27, 29, 44, 71, 76, 2, 17, 21, 24, 30, 31, 35, 40, 46, 49, 55, 67, 73,
            70, 3, 10, 12, 16, 42, 73, 1, 15, 25, 31, 37, 59, 70, 3, 8, 12, 16, 42,
            73, 62, 3, 8, 10, 16, 42, 73, 14, 31, 13, 31, 1, 9, 24, 25, 37, 39, 58,
            59, 70, 73, 3, 8, 10, 12, 42, 73, 2, 6, 21, 24, 30, 31, 35, 40, 46, 49,
            67, 34, 39, 45, 49, 51, 58, 70, 71, 72, 73, 75, 62, 62, 2, 6, 17, 24, 25,
            30, 31, 35, 40, 46, 49, 55, 67, 62, 5, 26, 27, 29, 44, 71, 76, 2, 6, 15,
            17, 21, 30, 31, 35, 39, 40, 46, 49, 55, 67, 73, 0, 1, 9, 15, 21, 37, 46,
            49, 58, 59, 70, 5, 23, 27, 29, 44, 71, 76, 3, 5, 23, 26, 29, 39, 44, 48,
            58, 65, 69, 70, 71, 73, 76, 36, 39, 60, 73, 5, 23, 26, 27, 44, 71, 76, 2,
            6, 17, 21, 24, 31, 35, 40, 46, 49, 67, 1, 2, 6, 9, 13, 14, 17, 21, 24,
            30, 35, 37, 39, 40, 46, 49, 53, 55, 59, 67, 70, 73, 62, 73, 4, 18, 45,
            47, 49, 51, 73, 2, 6, 17, 21, 24, 30, 31, 40, 55, 67, 28, 1, 9, 15, 25,
            31, 39, 58, 59, 70, 73, 73, 1, 3, 15, 18, 24, 27, 28, 31, 37, 58, 59, 69,
            70, 72, 73, 74, 75, 2, 6, 17, 21, 24, 30, 31, 35, 46, 49, 55, 67, 53, 3,
            8, 10, 12, 16, 73, 73, 5, 23, 26, 27, 29, 71, 76, 18, 34, 49, 51, 2, 6,
            17, 21, 24, 25, 30, 31, 40, 49, 61, 34, 58, 27, 73, 2, 4, 6, 17, 18, 21,
            24, 25, 30, 31, 34, 40, 45, 46, 51, 66, 70, 71, 73, 56, 62, 73, 18, 34,
            45, 49, 52, 57, 73, 51, 31, 41, 73, 2, 6, 21, 24, 31, 35, 40, 50, 62, 73,
            51, 66, 0, 1, 15, 18, 25, 27, 37, 39, 47, 70, 73, 1, 9, 15, 25, 31, 37,
            39, 70, 73, 28, 73, 46, 11, 19, 20, 22, 32, 50, 56, 63, 64, 73, 62, 62,
            27, 69, 49, 57, 70, 2, 6, 17, 21, 24, 30, 31, 35, 40, 73, 27, 39, 65, 73,
            0, 1, 7, 9, 15, 18, 25, 27, 31, 37, 39, 49, 58, 59, 66, 73, 5, 18, 23,
            26, 27, 29, 44, 49, 76, 18, 39, 73, 1, 3, 6, 8, 10, 12, 15, 16, 18, 24,
            27, 28, 31, 33, 34, 37, 38, 39, 42, 43, 48, 49, 50, 51, 54, 56, 58, 59,
            60, 62, 68, 69, 70, 72, 74, 75, 39, 73, 18, 39, 73, 5, 23, 26, 27, 29,
            44, 71,
        ],
        index_dtype,
    )
    weights = cp.array(
        [
            2, 1, 2, 3, 4, 1, 1, 6, 2, 1, 2, 6, 1, 4, 5, 6, 4, 3, 5, 1, 5, 2, 1, 1,
            2, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 3, 4, 3, 4, 4, 4, 3, 4, 9, 12, 10, 6, 5,
            3, 7, 1, 5, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 3, 1, 1, 1, 3, 1, 3, 2, 2, 2,
            2, 3, 3, 1, 1, 2, 2, 2, 2, 2, 3, 2, 3, 2, 4, 1, 1, 1, 4, 1, 1, 2, 4, 1,
            1, 2, 2, 2, 2, 2, 5, 9, 13, 15, 5, 6, 1, 5, 2, 5, 2, 3, 1, 1, 21, 2, 4,
            1, 1, 2, 31, 1, 2, 1, 6, 12, 13, 17, 1, 6, 7, 2, 5, 2, 9, 1, 3, 1, 3, 3,
            4, 5, 3, 3, 4, 4, 10, 1, 15, 17, 6, 7, 3, 6, 5, 1, 7, 1, 4, 4, 2, 1, 1,
            1, 1, 1, 1, 5, 2, 1, 3, 4, 3, 3, 3, 4, 4, 3, 1, 3, 4, 3, 4, 5, 3, 2, 2,
            1, 2, 1, 3, 9, 4, 2, 1, 3, 8, 4, 5, 3, 4, 3, 3, 4, 3, 6, 5, 6, 6, 2, 1,
            5, 1, 1, 2, 1, 5, 5, 1, 2, 2, 6, 7, 7, 2, 1, 1, 1, 3, 1, 4, 2, 1, 1, 1,
            1, 1, 1, 1, 1, 3, 1, 1, 12, 9, 2, 1, 3, 1, 2, 3, 1, 1, 2, 1, 1, 2, 6, 3,
            4, 1, 1, 1, 1, 2, 5, 1, 1, 2, 1, 1, 1, 6, 5, 1, 1, 1, 1, 1, 1, 5, 1, 17,
            1, 1, 5, 7, 5, 5, 5, 5, 3, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 2, 3, 1, 4, 3,
            4, 3, 3, 4, 3, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 1,
            1, 1, 5, 5, 21, 9, 7, 5, 1, 4, 12, 2, 1, 1, 6, 1, 2, 1, 19, 6, 8, 3, 2,
            9, 2, 6, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 10, 3, 1, 1, 1, 1,
            1, 4, 2, 2, 1, 1, 1, 13, 7, 2, 1, 2, 1, 1, 2, 1, 1, 1, 3, 1, 3, 1, 2, 1,
            1, 1, 8, 10, 1, 1, 5, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 3, 4, 2, 1, 1, 2, 1,
            2, 1, 2, 3, 2, 6, 1, 3, 4, 1, 3, 1, 1, 5, 5, 2, 13, 1, 1, 12, 4, 1, 3, 4,
            3, 3, 4, 1, 3, 2, 1, 1, 1, 2, 1, 2, 3, 2, 1, 2, 31, 4, 9, 8, 1, 1, 2, 1,
            1, 17, 3, 1, 1, 19, 3, 2, 1, 3, 7, 1, 1, 5, 1, 3, 12, 1, 2, 3, 1, 2, 1,
            1, 3, 3, 4, 3, 4, 4, 3, 3,
        ],
        np.int8,
    )
    nodes = [
        "Anzelma", "Babet", "Bahorel", "Bamatabois", "BaronessT", "Blacheville",
        "Bossuet", "Boulatruelle", "Brevet", "Brujon", "Champmathieu",
        "Champtercier", "Chenildieu", "Child1", "Child2", "Claquesous",
        "Cochepaille", "Combeferre", "Cosette", "Count", "CountessDeLo",
        "Courfeyrac", "Cravatte", "Dahlia", "Enjolras", "Eponine", "Fameuil",
        "Fantine", "Fauchelevent", "Favourite", "Feuilly", "Gavroche", "Geborand",
        "Gervais", "Gillenormand", "Grantaire", "Gribier", "Gueulemer", "Isabeau",
        "Javert", "Joly", "Jondrette", "Judge", "Labarre", "Listolier",
        "LtGillenormand", "Mabeuf", "Magnon", "Marguerite", "Marius",
        "MlleBaptistine", "MlleGillenormand", "MlleVaubois", "MmeBurgon", "MmeDeR",
        "MmeHucheloup", "MmeMagloire", "MmePontmercy", "MmeThenardier",
        "Montparnasse", "MotherInnocent", "MotherPlutarch", "Myriel", "Napoleon",
        "OldMan", "Perpetue", "Pontmercy", "Prouvaire", "Scaufflaire", "Simplice",
        "Thenardier", "Tholomyes", "Toussaint", "Valjean", "Woman1", "Woman2",
        "Zephine",
    ]
    # fmt: on
    use_compat_graph = _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs
    return nxcg.CudaGraph.from_coo(
        77,
        src_indices,
        dst_indices,
        edge_values={"weight": weights},
        id_to_key=nodes,
        use_compat_graph=use_compat_graph,
    )
