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
    offsets = None
    indices = None
    weights = None
    local_workers = None

    """
    We need a CSR of a symmetric graph.  Since we know it's symmetric
    either CSR or CSC will work the same.
    """
    if input_graph.adjlist is not None:
        if input_graph.adjlist.weights is None:
            raise Exception("hungarian algorithm requires weighted graph")

        offsets = input_graph.adjlist.offsets
        indices = input_graph.adjlist.indices
    elif input_graph.transposedadjlist is not None:
        if input_graph.tranposedadjlist.weights is None:
            raise Exception("hungarian algorithm requires weighted graph")

        offsets = input_graph.transposedadjlist.offsets
        indices = input_graph.transposedadjlist.indices
    else:
        if input_graph.edgelist.weights is None:
            raise Exception("hungarian algorithm requires weighted graph")

        input_graph.view_adj_list()
        offsets = input_graph.adjlist.offsets
        indices = input_graph.adjlist.indices

    [offsets, indices] = graph_new_wrapper.datatype_cast([offsets, indices],
                                                         [np.int32])

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
        [local_workers] = graph_new_wrapper.datatype_cast([offsets, indices],
                                                          [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    df = cudf.DataFrame()
    df['vertices'] = workers
    df['assignment'] = cudf.Series(np.zeros(len(workers), dtype=np.int32))
    
    cdef uintptr_t c_offsets        = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices        = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights        = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_workers        = local_workers.__cuda_array_interface__['data'][0]

    cdef uintptr_t c_identifier     = df['vertices'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_assignment     = df['assignment'].__cuda_array_interface__['data'][0];

    cdef GraphCSR[int,int,float] g_float
    cdef GraphCSR[int,int,double] g_double

    if weights.dtype == np.float32:
        g_float = GraphCSR[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, num_verts, num_edges)

        c_hungarian[int,int,float](g_float, len(workers), <int*>c_workers, <int*>c_assignment)
    else:
        g_double = GraphCSR[int,int,double](<int*>c_offsets, <int*>c_indices, <double*>c_weights, num_verts, num_edges)

        c_hungarian[int,int,double](g_double, len(workers), <int*>c_workers, <int*>c_assignment)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertices')

    return df
