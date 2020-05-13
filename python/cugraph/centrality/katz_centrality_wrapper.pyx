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

from cugraph.centrality.katz_centrality cimport katz_centrality as c_katz_centrality
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import rmm
import numpy as np


def get_output_df(input_graph, nstart):
    num_verts = input_graph.number_of_vertices()
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    if nstart is None:
        df['katz_centrality'] = cudf.Series(np.zeros(num_verts, dtype=np.float64))
    else:
        if len(nstart) != num_verts:
            raise ValueError('nstart must have initial guess for all vertices')

        nstart = graph_wrapper.datatype_cast([nstart], [np.float64])

        if input_graph.renumbered is True:
            renumber_series = cudf.Series(input_graph.edgelist.renumber_map.index,
                                          index=input_graph.edgelist.renumber_map)
            nstart_vertex_renumbered = cudf.Series(renumber_series.loc[nstart['vertex']], dtype=np.int32)
            df['katz_centrality'] = cudf.Series(cudf._lib.copying.scatter(nstart['values']._column,
                                                nstart_vertex_renumbered._column,
                                                df['katz_centrality']._column))
        else:
            df['katz_centrality'] = cudf.Series(cudf._lib.copying.scatter(nstart['values']._column,
                                                nstart['vertex']._column,
                                                df['katz_centrality']._column))
    return df


def katz_centrality(input_graph, alpha=None, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    """
    Call katz_centrality
    """

    df = get_output_df(input_graph, nstart)
    if nstart is not None:
        has_guess = True
    if alpha is None:
        alpha = 0

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_katz = df['katz_centrality'].__cuda_array_interface__['data'][0]

    cdef GraphCSRViewFloat graph = get_graph_view[GraphCSRViewFloat](input_graph, True)

    c_katz_centrality[int,int,float,double](graph, <double*> c_katz, alpha, max_iter, tol, has_guess, normalized)

    graph.get_vertex_identifiers(<int*>c_identifier)

    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')

    return df
