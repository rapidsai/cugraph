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

from cugraph.components.c_connectivity cimport *
from cugraph.structure.c_graph cimport *
from cugraph.structure.graph_wrapper cimport *
from libc.stdint cimport uintptr_t
import cudf
import numpy as np

cpdef weakly_connected_components(graph_ptr, connect_type = CUGRAPH_WEAK):
    """
    Call gdf_connected_components
    """

    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_adj_list(<gdf_graph*> graph)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    # we should add get_number_of_vertices() to gdf_graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['labels'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_labels = get_gdf_column_view(df['labels'])

    err = gdf_connected_components(g, <cugraph_cc_t>connect_type, &c_labels)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df
