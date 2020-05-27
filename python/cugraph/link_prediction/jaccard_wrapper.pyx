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

from cugraph.link_prediction.jaccard cimport jaccard as c_jaccard
from cugraph.link_prediction.jaccard cimport jaccard_list as c_jaccard_list
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from libc.stdint cimport uintptr_t
from cython cimport floating

import cudf
import numpy as np


def jaccard(input_graph, weights_arr=None, vertex_pair=None):
    """
    Call jaccard or jaccard_list
    """
    offsets = None
    indices = None

    if input_graph.adjlist:
        [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.adjlist.offsets,
                                                              input_graph.adjlist.indices], [np.int32])
    elif input_graph.transposedadjlist:
        #
        # NOTE: jaccard ONLY operates on an undirected graph, so CSR and CSC should be
        #       equivalent.  The undirected check has already happened, so we'll just use
        #       the CSC as if it were CSR.
        #
        [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.transposedadjlist.offsets,
                                                              input_graph.transposedadjlist.indices], [np.int32])
    else:
        input_graph.view_adj_list()
        [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.adjlist.offsets,
                                                              input_graph.adjlist.indices], [np.int32])
        
    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    cdef uintptr_t c_result_col = <uintptr_t> NULL
    cdef uintptr_t c_first_col = <uintptr_t> NULL
    cdef uintptr_t c_second_col = <uintptr_t> NULL
    cdef uintptr_t c_src_index_col = <uintptr_t> NULL
    cdef uintptr_t c_dst_index_col = <uintptr_t> NULL
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double

    weight_type = np.float32

    if weights_arr is not None:
        [weights] = graph_new_wrapper.datatype_cast([weights_arr], [np.float32, np.float64])
        c_weights = weights.__cuda_array_interface__['data'][0]
        weight_type = weights.dtype

    if type(vertex_pair) == cudf.DataFrame:
        result_size = len(vertex_pair)
        result = cudf.Series(np.ones(result_size, dtype=weight_type))
        c_result_col = result.__cuda_array_interface__['data'][0]

        df = cudf.DataFrame()
        df['jaccard_coeff'] = result

        if input_graph.renumbered is True:
            renumber_df = cudf.DataFrame()
            renumber_df['map'] = input_graph.edgelist.renumber_map
            renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
            vp = vertex_pair.merge(renumber_df, left_on='first', right_on='map', how='left').drop('map').merge(renumber_df, left_on='second', right_on='map', how='left').drop('map')

            df['source'] = vp['first']
            df['destination'] = vp['second']
            c_first_col = vp['id_x'].__cuda_array_interface__['data'][0]
            c_second_col = vp['id_y'].__cuda_array_interface__['data'][0]
        else:
            cols = vertex_pair.columns.to_list()
            first = vertex_pair[cols[0]].astype(np.int32)
            second = vertex_pair[cols[1]].astype(np.int32)
            df['source'] = first
            df['destination'] = second
            c_first_col = first.__cuda_array_interface__['data'][0]
            c_second_col = second.__cuda_array_interface__['data'][0]


        if weight_type == np.float32:
            graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                                  <float*>c_weights, num_verts, num_edges)
            c_jaccard_list[int,int,float](graph_float,
                                          <float*>c_weights,
                                          result_size,
                                          <int*>c_first_col,
                                          <int*>c_second_col,
                                          <float*>c_result_col)
        else:
            graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                    <double*>c_weights, num_verts, num_edges)
            c_jaccard_list[int,int,double](graph_double,
                                           <double*>c_weights,
                                           result_size,
                                           <int*>c_first_col,
                                           <int*>c_second_col,
                                           <double*>c_result_col)
            
        return df
    else:
        # error check performed in jaccard.py
        assert vertex_pair is None

        df = cudf.DataFrame()
        df['source'] = cudf.Series(np.zeros(num_edges, indices.dtype))
        df['destination'] = indices

        c_src_index_col = df['source'].__cuda_array_interface__['data'][0]

        if weight_type == np.float32:
            df['jaccard_coeff'] = cudf.Series(np.ones(num_edges, dtype=np.float32),
                                              nan_as_null=False)
            c_result_col = df['jaccard_coeff'].__cuda_array_interface__['data'][0]

            graph_float = GraphCSRView[int,int,float](<int*>c_offsets,
                                                  <int*>c_indices,
                                                  <float*>c_weights,
                                                  num_verts,
                                                  num_edges)
            c_jaccard[int,int,float](graph_float,
                                     <float*>c_weights,
                                     <float*>c_result_col)

            graph_float.get_source_indices(<int*>c_src_index_col)
        else:
            df['jaccard_coeff'] = cudf.Series(np.ones(num_edges, dtype=np.float64),
                                              nan_as_null=False)
            c_result_col = df['jaccard_coeff'].__cuda_array_interface__['data'][0]

            graph_double = GraphCSRView[int,int,double](<int*>c_offsets,
                                                    <int*>c_indices,
                                                    <double*>c_weights,
                                                    num_verts,
                                                    num_edges)
            c_jaccard[int,int,double](graph_double,
                                      <double*>c_weights,
                                      <double*>c_result_col)

            graph_double.get_source_indices(<int*>c_src_index_col)
            
        if input_graph.renumbered:
            if isinstance(input_graph.edgelist.renumber_map, cudf.DataFrame):
                unrenumbered_df_ = df.merge(input_graph.edgelist.renumber_map, left_on='source', right_on='id', how='left').drop(['id', 'source'])
                unrenumbered_df = unrenumbered_df_.merge(input_graph.edgelist.renumber_map, left_on='destination', right_on='id', how='left').drop(['id', 'destination'])
                cols = unrenumbered_df.columns.to_list()
                df = unrenumbered_df[cols[1:] + [cols[0]]]
            else:
                df['source'] = input_graph.edgelist.renumber_map[df['source']].reset_index(drop=True)
                df['destination'] = input_graph.edgelist.renumber_map[df['destination']].reset_index(drop=True)

        return df
