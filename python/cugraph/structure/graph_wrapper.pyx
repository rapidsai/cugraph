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

from cugraph.structure.graph cimport *
cimport cugraph.structure.graph
from cugraph.structure.graph cimport get_two_hop_neighbors as c_get_two_hop_neighbors
from cugraph.structure.graph cimport renumber_vertices as c_renumber_vertices
from cugraph.structure.utils_wrapper import *
from libcpp cimport bool
from libc.stdint cimport uintptr_t

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer

import dask_cudf as dc
from cugraph.raft.dask.common.comms import Comms
from dask.distributed import wait, default_client
from cugraph.raft.dask.common.comms import worker_state
from cugraph.dask.common.input_utils import DistributedDataHandler

import cudf
from cudf.core.buffer import Buffer
import rmm
import numpy as np


def datatype_cast(cols, dtypes):
    cols_out = []
    for col in cols:
        if col is None or col.dtype.type in dtypes:
            cols_out.append(col)
        else:
            cols_out.append(col.astype(dtypes[0]))
    return cols_out


def renumber(source_col, dest_col):
    num_edges = len(source_col)
    
    src_renumbered = cudf.Series(np.zeros(num_edges), dtype=np.int32)
    dst_renumbered = cudf.Series(np.zeros(num_edges), dtype=np.int32)

    cdef uintptr_t c_src = source_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dest_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_src_renumbered = src_renumbered.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_renumbered = dst_renumbered.__cuda_array_interface__['data'][0]
    cdef int map_size = 0
    cdef int n_edges = num_edges

    cdef unique_ptr[device_buffer] numbering_map

    if (source_col.dtype == np.int32):
        numbering_map = move(c_renumber_vertices[int,int,int](n_edges,
                                                              <int*>c_src,
                                                              <int*>c_dst,
                                                              <int*>c_src_renumbered,
                                                              <int*>c_dst_renumbered,
                                                              &map_size))
    else:
        numbering_map = move(c_renumber_vertices[long,int,int](n_edges,
                                                               <long*>c_src,
                                                               <long*>c_dst,
                                                               <int*>c_src_renumbered,
                                                               <int*>c_dst_renumbered,
                                                               &map_size))
        
        
    map = DeviceBuffer.c_from_unique_ptr(move(numbering_map))
    map = Buffer(map)
    
    output_map = cudf.Series(data=map, dtype=source_col.dtype)

    return src_renumbered, dst_renumbered, output_map
                                        

def view_adj_list(input_graph):

    if input_graph.adjlist is None:
        if input_graph.edgelist is None:
            raise Exception('Graph is Empty')

        [src, dst] = datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        weights = None
        if input_graph.edgelist.weights:
            [weights] = datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])

        return coo2csr(src, dst, weights)


def view_transposed_adj_list(input_graph):

    if input_graph.transposedadjlist is None:
        if input_graph.edgelist is None:
            if input_graph.adjlist is None:
                raise Exception('Graph is Empty')
            else:
                input_graph.view_edge_list()

        [src, dst] = datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        weights = None
        if input_graph.edgelist.weights:
            [weights] = datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])

        return coo2csr(dst, src, weights)


def view_edge_list(input_graph):

    if input_graph.adjlist is None:
        raise Exception('Graph is Empty')

    [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef GraphCSRView[int,int,float] graph
    graph = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    src_indices = cudf.Series(np.zeros(num_edges), dtype= indices.dtype)
    cdef uintptr_t c_src_indices = src_indices.__cuda_array_interface__['data'][0]
    graph.get_source_indices(<int*>c_src_indices)

    return src_indices, indices, weights


def _degree_coo(edgelist_df, src_name, dst_name, x=0, num_verts=None, sID=None):
    #
    #  Computing the degree of the input graph from COO
    #
    cdef DegreeDirection dir

    src = edgelist_df[src_name]
    dst = edgelist_df[dst_name]

    if x == 0:
        dir = DIRECTION_IN_PLUS_OUT
    elif x == 1:
        dir = DIRECTION_IN
    elif x == 2:
        dir = DIRECTION_OUT
    else:
        raise Exception("x should be 0, 1 or 2")

    [src, dst] = datatype_cast([src, dst], [np.int32])

    if num_verts is None:
        num_verts = 1 + max(src.max(), dst.max())
    num_edges = len(src)

    vertex_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    degree_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef GraphCOOView[int,int,float] graph

    cdef uintptr_t c_vertex = vertex_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_degree = degree_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_src = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dst.__cuda_array_interface__['data'][0]

    graph = GraphCOOView[int,int,float](<int*>c_src, <int*>c_dst, <float*>NULL, num_verts, num_edges)

    cdef size_t handle_size_t
    if sID is not None:
        sessionstate = worker_state(sID)
        print("nworkers: ", sessionstate['nworkers'],"  id: ", sessionstate['wid'])
        handle = sessionstate['handle']
        handle_size_t = <size_t>handle.getHandle()
        graph.set_handle(<handle_t*>handle_size_t)

    graph.degree(<int*> c_degree, dir)
    graph.get_vertex_identifiers(<int*>c_vertex)

    return vertex_col, degree_col


def _degree_csr(offsets, indices, x=0):
    cdef DegreeDirection dir

    if x == 0:
        dir = DIRECTION_IN_PLUS_OUT
    elif x == 1:
        dir = DIRECTION_IN
    elif x == 2:
        dir = DIRECTION_OUT
    else:
        raise Exception("x should be 0, 1 or 2")

    [offsets, indices] = datatype_cast([offsets, indices], [np.int32])

    num_verts = len(offsets)-1
    num_edges = len(indices)

    vertex_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    degree_col = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef GraphCSRView[int,int,float] graph

    cdef uintptr_t c_vertex = vertex_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_degree = degree_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]

    graph = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)
        
    graph.degree(<int*> c_degree, dir)
    graph.get_vertex_identifiers(<int*>c_vertex)

    return vertex_col, degree_col


def _degree(input_graph, x=0):
    transpose_x = { 0: 0,
                    2: 1,
                    1: 2 }
    
    if input_graph.adjlist is not None:
        return _degree_csr(input_graph.adjlist.offsets,
                           input_graph.adjlist.indices,
                           x)

    if input_graph.transposedadjlist is not None:
        return _degree_csr(input_graph.transposedadjlist.offsets,
                           input_graph.transposedadjlist.indices,
                           transpose_x[x])

    if input_graph.edgelist is not None:
        if isinstance(input_graph.edgelist.edgelist_df, dc.DataFrame):
            input_ddf = input_graph.edgelist.edgelist_df
            cols = input_ddf.columns
            num_verts = input_ddf[cols[0:2]].max().max().compute() + 1
            data = DistributedDataHandler.create(data=input_ddf)
            comms = Comms(comms_p2p=False)
            comms.init()
            #degree_ddf = input_ddf.map_partitions(_degree_coo, input_ddf.columns[0], input_ddf.columns[1], x, num_verts, comms.sessionId)
            client = default_client()
            #data = DistributedDataHandler.create(data=ddf)
            data.calculate_parts_to_sizes(comms)
            degree_ddf = [client.submit(_degree_coo, wf[1][0], cols[0], cols[1], x, num_verts, comms.sessionId, workers=[wf[0]]) for idx, wf in enumerate(data.worker_to_parts.items())]
            wait(degree_ddf)
            return degree_ddf[0].result()      
        return _degree_coo(input_graph.edgelist.edgelist_df,
                           'src', 'dst', x)
                           
    raise Exception("input_graph not COO, CSR or CSC")

    
def _degrees(input_graph):
    verts, indegrees = _degree(input_graph,1)
    verts, outdegrees = _degree(input_graph, 2)
    
    return verts, indegrees, outdegrees


def p_get_two_hop_neighbors(input_graph):
    cdef GraphCSRView[int,int,float] graph

    offsets = None
    indices = None
    transposed = False

    if input_graph.adjlist:
        [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    elif input_graph.transposedadjlist:
        [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
        transposed = True
    else:
        input_graph.view_adj_list()
        [offsets, indices] = datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    graph = GraphCSRView[int,int,float](<int*>c_offsets, <int*> c_indices, <float*>NULL, num_verts, num_edges)

    df = coo_to_df(move(c_get_two_hop_neighbors(graph)))
    if not transposed:
        df.rename(columns={'src':'first', 'dst':'second'}, inplace=True)
    else:
        df.rename(columns={'dst':'first', 'src':'second'}, inplace=True)

    return df


def weight_type(input_graph):
    weights_type = None
    if input_graph.edgelist.weights:
        weights_type = input_graph.edgelist.edgelist_df['weights'].dtype
    return weights_type


cdef coo_to_df(GraphCOOPtrType graph):
    contents = move(graph.get()[0].release())
    src = DeviceBuffer.c_from_unique_ptr(move(contents.src_indices))
    dst = DeviceBuffer.c_from_unique_ptr(move(contents.dst_indices))
    wgt = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    src = Buffer(src)
    dst = Buffer(dst)
    wgt = Buffer(wgt)

    src = cudf.Series(data=src, dtype="int32")
    dst = cudf.Series(data=dst, dtype="int32")

    df = cudf.DataFrame()
    df['src'] = src
    df['dst'] = dst
    if wgt.nbytes != 0:
        if GraphCOOPtrType is GraphCOOPtrFloat:
            wgt = cudf.Series(data=wgt, dtype="float32")
        elif GraphCOOPtrType is GraphCOOPtrDouble:
            wgt = cudf.Series(data=wgt, dtype="float64")
        df['weight'] = wgt
    return df


##  TODO:  Move this back into a separate file...
cdef csr_to_series(GraphCSRPtrType graph):
    contents = move(graph.get()[0].release())
    offsets = DeviceBuffer.c_from_unique_ptr(move(contents.offsets))
    indices = DeviceBuffer.c_from_unique_ptr(move(contents.indices))
    weights = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    offsets = Buffer(offsets)
    indices = Buffer(indices)
    weights = Buffer(weights)

    csr_offsets = cudf.Series(data=offsets, dtype="int32")
    csr_indices = cudf.Series(data=indices, dtype="int32")

    csr_weights = None
    if weights.nbytes != 0:
        if GraphCSRPtrType is GraphCSRPtrFloat:
            csr_weights = cudf.Series(data=weights, dtype="float32")
        elif GraphCSRPtrType is GraphCSRPtrDouble:
            csr_weights = cudf.Series(data=weights, dtype="float64")
    return (csr_offsets, csr_indices, csr_weights)


"""
cdef GraphCSRViewType get_csr_graph_view(input_graph, bool weighted=True, GraphCSRViewType* dummy=NULL):
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    cdef uintptr_t c_off = input_graph.adjlist.offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_ind = input_graph.adjlist.indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if input_graph.adjlist.weights and weighted:
        c_weights = input_graph.adjlist.weights.__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    cdef GraphCSRViewType in_graph
    if GraphCSRViewType is GraphCSRViewFloat:
        in_graph = GraphCSRViewFloat(<int*>c_off, <int*>c_ind, <float*>c_weights, num_verts, num_edges)
    elif GraphCSRViewType is GraphCSRViewDouble:
        in_graph = GraphCSRViewDouble(<int*>c_off, <int*>c_ind, <double*>c_weights, num_verts, num_edges)
    return in_graph


cdef GraphCOOViewType get_coo_graph_view(input_graph, bool weighted=True, GraphCOOViewType* dummy=NULL):
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    cdef uintptr_t c_src = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if input_graph.edgelist.weights and weighted:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    cdef GraphCOOViewType in_graph
    if GraphCOOViewType is GraphCOOViewFloat:
        in_graph = GraphCOOViewFloat(<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
    elif GraphCOOViewType is GraphCOOViewDouble:
        in_graph = GraphCOOViewDouble(<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
    return in_graph


cdef GraphViewType get_graph_view(input_graph, bool weighted = True, GraphViewType* dummy=NULL):
    if GraphViewType is GraphCOOViewFloat:
        return get_coo_graph_view[GraphCOOViewFloat](input_graph, weighted, dummy)
    elif GraphViewType is GraphCOOViewDouble:
        return get_coo_graph_view[GraphCOOViewDouble](input_graph, weighted, dummy)
    elif GraphViewType is GraphCSRViewFloat:
        return get_csr_graph_view[GraphCSRViewFloat](input_graph, weighted, dummy)
    elif GraphViewType is GraphCSRViewDouble:
        return get_csr_graph_view[GraphCSRViewDouble](input_graph, weighted, dummy)
"""
