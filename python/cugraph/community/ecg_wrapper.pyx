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

cimport cugraph.community.ecg as c_ecg
from cugraph.structure.graph cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import rmm
import numpy as np


def ecg(input_graph, min_weight=.05, ensemble_size=16):
    """
    Call ECG
    """

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    if input_graph.adjlist.weights is None:
        raise Exception('ECG must be called on a weighted graph')

    [offsets, indices] = graph_wrapper.datatype_cast([input_graph.adjlist.offsets,
                                                      input_graph.adjlist.indices], [np.int32, np.int64])
    [weights] = graph_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])

    cdef uintptr_t graph = graph_wrapper.allocate_cpp_graph()
    cdef Graph * g = <Graph*> graph

    graph_wrapper.add_adj_list(graph, offsets, indices, weights)

    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_index_col = get_gdf_column_view(df['vertex'])
    g.adjList.get_vertex_identifiers(&c_index_col)

    df['partition'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef uintptr_t c_ecg_ptr = df['partition'].__cuda_array_interface__['data'][0]

    if weights.dtype == np.float32:
        c_ecg.ecg[int32_t, float] (<Graph*>g, min_weight, ensemble_size, <int32_t*>c_ecg_ptr)
    else:
        c_ecg.ecg[int32_t, double] (<Graph*>g, min_weight, ensemble_size, <int32_t*>c_ecg_ptr)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')

    return df
    
