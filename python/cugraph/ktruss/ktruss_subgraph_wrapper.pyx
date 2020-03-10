# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.ktruss.ktruss_subgraph cimport *
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def ktruss_subgraph(input_graph, k, subgraph_truss):
    """
    Call ktruss
    """
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges()

    cdef GraphCOO[int,int,float] input_coo
    cdef GraphCOO[int,int,float] output_coo

    cdef uintptr_t c_src_indices = input_graph.edgelist.source.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.dest.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    input_coo = GraphCOO[int,int,float](<int*>c_src_indices, <int*>c_dst_indices, <float*>c_weights, num_verts, num_edges)
    output_coo = GraphCOO[int,int,float]()
    k_truss_subgraph(input_coo, k, output_coo);

    src_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.src_indices,
            nelem=output_coo.number_of_edges,
            dtype=np.int32)

    dst_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.dst_indices,
            nelem=output_coo.number_of_edges,
            dtype=np.int32)
    df = cudf.DataFrame()
    df['src'] = cudf.Series(src_array)
    df['dst'] = cudf.Series(dst_array)

    if input_graph.renumbered:
        unrenumber(input_graph.edgelist.renumber_map, df, 'src')
        unrenumber(input_graph.edgelist.renumber_map, df, 'dst')
    subgraph_truss.from_cudf_edgelist(df, source='src', destination='dst', renumber=False)
