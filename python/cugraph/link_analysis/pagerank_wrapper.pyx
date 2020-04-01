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

#cimport cugraph.link_analysis.pagerank as c_pagerank
from cugraph.link_analysis.pagerank cimport pagerank as c_pagerank
from cugraph.structure.graph_new cimport *
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from cugraph.structure import graph_wrapper
import cudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib


def pagerank(input_graph, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-5, nstart=None):
    """
    Call pagerank
    """

    if not input_graph.transposedadjlist:
        input_graph.view_transposed_adj_list()

    [offsets, indices] = graph_wrapper.datatype_cast([input_graph.transposedadjlist.offsets, input_graph.transposedadjlist.indices], [np.int32])
    [weights] = graph_wrapper.datatype_cast([input_graph.transposedadjlist.weights], [np.float32, np.float64])

    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef bool has_guess = <bool> 0
    if nstart is not None:
        if len(nstart) != num_verts:
            raise ValueError('nstart must have initial guess for all vertices')
        if input_graph.renumbered is True:
            renumber_df = cudf.DataFrame()
            renumber_df['map'] = input_graph.edgelist.renumber_map
            renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
            guess = nstart.merge(renumber_df, left_on='vertex', right_on='map', how='left').drop('map')
            df['pagerank'][guess['id']] = guess['values']
        else:
            df['pagerank'][nstart['vertex']] = nstart['values']
        has_guess = <bool> 1

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_pagerank_val = df['pagerank'].__cuda_array_interface__['data'][0];

    cdef uintptr_t c_pers_vtx = <uintptr_t>NULL
    cdef uintptr_t c_pers_val = <uintptr_t>NULL
    cdef sz = 0

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCSC[int,int,float] graph_float
    cdef GraphCSC[int,int,double] graph_double
    
    if personalization is not None:
        sz = personalization['vertex'].shape[0]
        personalization['vertex'] = personalization['vertex'].astype(np.int32)
        personalization['values'] = personalization['values'].astype(df['pagerank'].dtype)
        if input_graph.renumbered is True:
            renumber_df = cudf.DataFrame()
            renumber_df['map'] = input_graph.edgelist.renumber_map
            renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
            personalization_values = personalization.merge(renumber_df, left_on='vertex', right_on='map', how='left').drop('map')
            c_pers_vtx = personalization_values['id'].__cuda_array_interface__['data'][0]
            c_pers_val = personalization_values['values'].__cuda_array_interface__['data'][0]
        else:
            c_pers_vtx = personalization['vertex'].__cuda_array_interface__['data'][0]
            c_pers_val = personalization['values'].__cuda_array_interface__['data'][0]
    
    if (df['pagerank'].dtype == np.float32): 
        graph_float = GraphCSC[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, num_verts, num_edges)

        c_pagerank[int,int,float](graph_float, <float*> c_pagerank_val, sz, <int*> c_pers_vtx, <float*> c_pers_val,
                               <float> alpha, <float> tol, <int> max_iter, has_guess)
        graph_float.get_vertex_identifiers(<int*>c_identifier)
    else: 
        graph_double = GraphCSC[int,int,double](<int*>c_offsets, <int*>c_indices, <double*>c_weights, num_verts, num_edges)
        c_pagerank[int,int,double](graph_double, <double*> c_pagerank_val, sz, <int*> c_pers_vtx, <double*> c_pers_val,
                            <float> alpha, <float> tol, <int> max_iter, has_guess)
        graph_double.get_vertex_identifiers(<int*>c_identifier)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')

    return df
