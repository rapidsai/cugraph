# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.matching.matching cimport hungarian as c_hungarian
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from libc.stdint cimport uintptr_t
from cugraph.structure.graph import Graph as type_Graph
from cugraph.utilities.unrenumber import unrenumber

import cudf
import numpy as np

def hungarian(input_graph, workers):
    """
    Call the hungarian algorithm
    """
    src = None
    dst = None
    weights = None
    local_workers = None

    """
    We need a COO of the graph.
    """
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    if input_graph.edgelist.weights is None:
        raise Exception("hungarian algorithm requires weighted graph")

    src = input_graph.edgelist.edgelist_df['src']
    dst = input_graph.edgelist.edgelist_df['dst']

    [src, dst] = graph_new_wrapper.datatype_cast([src, dst], [np.int32])
    [weights] = graph_new_wrapper.datatype_cast([weights], [np.float32, np.float64])

    """
    Need to renumber the workers
    """
    if input_graph.renumbered is True:
        renumber_df = cudf.DataFrame()
        renumber_df['map'] = input_graph.edgelist.renumber_map
        renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
        workers_df = cudf.DataFrame()
        workers_df['vertex'] = workers
        local_workers = workers_df.merge(renumber_df, left_on='vertex', right_on='map', how='left')['id']
    else:
        [local_workers] = graph_new_wrapper.datatype_cast([workers], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = len(src)

    df = cudf.DataFrame()
    df['vertices'] = workers
    df['assignment'] = cudf.Series(np.zeros(len(workers), dtype=np.int32))
    
    cdef uintptr_t c_src        = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst        = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights    = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_workers    = local_workers.__cuda_array_interface__['data'][0]

    cdef uintptr_t c_identifier = df['vertices'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_assignment = df['assignment'].__cuda_array_interface__['data'][0];

    cdef GraphCOO[int,int,float] g_float
    cdef GraphCOO[int,int,double] g_double

    if weights.dtype == np.float32:
        g_float = GraphCOO[int,int,float](<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)

        c_hungarian[int,int,float](g_float, len(workers), <int*>c_workers, <int*>c_assignment)
    else:
        g_double = GraphCOO[int,int,double](<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)

        c_hungarian[int,int,double](g_double, len(workers), <int*>c_workers, <int*>c_assignment)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertices')

    return df
