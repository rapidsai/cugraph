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

def mg_pagerank(input_df, local_data, rank, handle, alpha=0.85, max_iter=100, tol=1.0e-5, personalization=None, nstart=None):
    """
    Call pagerank
    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <c_pagerank.handle_t*>handle_size_t


    src = input_df['src']
    dst = input_df['dst']

    num_verts = local_data['verts'].sum()
    num_edges = local_data['edges'].sum()

    local_offset = local_data['offsets'][rank]
    dst = dst - local_offset
    num_local_verts = local_data['verts'][rank]
    num_local_edges = len(src)

    cdef uintptr_t c_local_verts = local_data['verts'].__array_interface__['data'][0]
    cdef uintptr_t c_local_edges = local_data['edges'].__array_interface__['data'][0]
    cdef uintptr_t c_local_offsets = local_data['offsets'].__array_interface__['data'][0]

    [src, dst] = graph_primtypes_wrapper.datatype_cast([src, dst], [np.int32])
    _offsets, indices, weights = coo2csr(dst, src, None)
    offsets = _offsets[:num_local_verts + 1]
    del _offsets
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_pagerank_val = df['pagerank'].__cuda_array_interface__['data'][0];

    cdef uintptr_t c_pers_vtx = <uintptr_t>NULL
    cdef uintptr_t c_pers_val = <uintptr_t>NULL
    cdef int sz = 0

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    #replace graph view construction w/ `cdef graph_container_t graph_container`
    #(see louvain_wrapper.pyx lines 97, 102)
    #
    weightTypeMap = {np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    cdef graph_container_t graph_container

    # FIXME: The excessive casting for the enum arg is needed to make cython
    #        understand how to pass the enum value (this is the same pattern
    #        used by cudf). This will not be needed with Cython 3.0
    populate_graph_container(graph_container,
                             <legacyGraphTypeEnum>(<int>(legacyGraphTypeEnum.CSR)),
                             handle_[0],
                             <void*>c_offsets, <void*>c_indices, <void*>c_weights,
                             <numberTypeEnum>(<int>(numberTypeEnum.intType)),
                             <numberTypeEnum>(<int>(numberTypeEnum.intType)),
                             <numberTypeEnum>(<int>(weightTypeMap[weights.dtype])),
                             num_verts, num_local_edges,
                             <int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets,
                             False, True)  # store_transposed, multi_gpu

    # Old code:
    #
    # cdef GraphCSCView[int,int,float] graph_float
    # cdef GraphCSCView[int,int,double] graph_double

    if personalization is not None:
        sz = personalization['vertex'].shape[0]
        personalization['vertex'] = personalization['vertex'].astype(np.int32)
        personalization['values'] = personalization['values'].astype(df['pagerank'].dtype)
        c_pers_vtx = personalization['vertex'].__cuda_array_interface__['data'][0]
        c_pers_val = personalization['values'].__cuda_array_interface__['data'][0]

    # after populate_graph_container()
    # pass the `graph_container` to c_pagerank.call_pagerank() instead
    # maybe drop 2 template parameter and only keep `WT` (see Louvain)
    #
    if weights.dtype == np.float32:
        c_pagerank.call_pagerank[int, float](handle_[0], graph_container, <float*> c_pagerank_val, sz, <int*> c_pers_vtx, <float*> c_pers_val,
                               <float> alpha, <float> tol, <int> max_iter, <bool> 0)
        # graph_float.get_vertex_identifiers(<int*>c_identifier) # <- TODO (how?)

    else:
        c_pagerank.call_pagerank[int, double](handle_[0], graph_container, <double*> c_pagerank_val, sz, <int*> c_pers_vtx, <double*> c_pers_val,
                            <float> alpha, <float> tol, <int> max_iter, <bool> 0)
        # graph_double.get_vertex_identifiers(<int*>c_identifier) # <- TODO (how?)


    # Old code:
    #
    # if (df['pagerank'].dtype == np.float32): # might not need that with `graph_container`
        
    #     graph_float = GraphCSCView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>c_weights, num_verts, num_local_edges)
    #     graph_float.set_local_data(<int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)
    #     graph_float.set_handle(handle_)
    #     c_pagerank.pagerank[int,int,float](handle_[0], graph_float, <float*> c_pagerank_val, sz, <int*> c_pers_vtx, <float*> c_pers_val,
    #                            <float> alpha, <float> tol, <int> max_iter, <bool> 0)
    #     graph_float.get_vertex_identifiers(<int*>c_identifier)
    # else:
    #     graph_double = GraphCSCView[int,int,double](<int*>c_offsets, <int*>c_indices, <double*>c_weights, num_verts, num_local_edges)
    #     graph_double.set_local_data(<int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)
    #     graph_double.set_handle(handle_)
    #     c_pagerank.pagerank[int,int,double](handle_[0], graph_double, <double*> c_pagerank_val, sz, <int*> c_pers_vtx, <double*> c_pers_val,
    #                         <float> alpha, <float> tol, <int> max_iter, <bool> 0)
    #     graph_double.get_vertex_identifiers(<int*>c_identifier)

    return df
