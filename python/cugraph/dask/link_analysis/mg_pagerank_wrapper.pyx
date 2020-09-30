#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
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
#

from cugraph.structure.utils_wrapper import *
from cugraph.dask.link_analysis cimport mg_pagerank as c_pagerank
import cudf
from cugraph.structure.graph_primtypes cimport *
import cugraph.structure.graph_primtypes_wrapper as graph_primtypes_wrapper
from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

def mg_pagerank(input_df,
                num_global_verts,
                num_global_edges,
                partition_row_size,
                partition_col_size,
                vertex_partition_offsets,
                rank, handle, alpha=0.85, max_iter=100, tol=1.0e-5,
                personalization=None, nstart=None):
    """
    Call pagerank
    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <c_pagerank.handle_t*>handle_size_t


    src = input_df['src']
    dst = input_df['dst']
    if "value" in input_df.columns:
        weights = input_df['value']
    else:
        weights = None

    # FIXME: needs to be edge_t type not int
    cdef int num_partition_edges = len(src)

    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL

    # FIXME: data is on device, move to host (to_pandas()), convert to np array and access pointer to pass to C
    cdef uintptr_t c_vertex_partition_offsets = vertex_partition_offsets.values_host.__array_interface__['data'][0]

    cdef graph_container_t graph_container

    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>c_vertex_partition_offsets,
                             <numberTypeEnum>(<int>(numberTypeEnum.intType)),
                             <numberTypeEnum>(<int>(numberTypeEnum.intType)),
                             <numberTypeEnum>(<int>(weightTypeMap[weights.dtype])),
                             num_partition_edges,
                             num_global_verts, num_global_edges,
                             partition_row_size, partition_col_size,
                             False, True) 

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_pagerank_val = df['pagerank'].__cuda_array_interface__['data'][0];

    cdef uintptr_t c_pers_vtx = <uintptr_t>NULL
    cdef uintptr_t c_pers_val = <uintptr_t>NULL
    cdef int sz = 0

    cdef GraphCSCView[int,int,float] graph_float
    cdef GraphCSCView[int,int,double] graph_double

    if personalization is not None:
        sz = personalization['vertex'].shape[0]
        personalization['vertex'] = personalization['vertex'].astype(np.int32)
        personalization['values'] = personalization['values'].astype(df['pagerank'].dtype)
        c_pers_vtx = personalization['vertex'].__cuda_array_interface__['data'][0]
        c_pers_val = personalization['values'].__cuda_array_interface__['data'][0]

    if (df['pagerank'].dtype == np.float32):
        c_pagerank.call_pagerank[int, float](handle_[0], graph_container, <float*> c_pagerank_val, sz, <int*> c_pers_vtx, <float*> c_pers_val,
                                 <float> alpha, <float> tol, <int> max_iter, <bool> 0)
    else:
        c_pagerank.call_pagerank[int, double](handle_[0], graph_container, <double*> c_pagerank_val, sz, <int*> c_pers_vtx, <double*> c_pers_val,
                            <float> alpha, <float> tol, <int> max_iter, <bool> 0)

    return df
