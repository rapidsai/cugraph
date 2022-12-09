# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.structure.graph_primtypes cimport *
from cugraph.structure.graph_primtypes cimport get_two_hop_neighbors as c_get_two_hop_neighbors
from cugraph.structure.utils_wrapper import *
from libcpp cimport bool
import enum
from libc.stdint cimport uintptr_t

import dask_cudf as dc
import cugraph.dask.comms.comms as Comms
from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import DistributedDataHandler

import cudf
import numpy as np


def datatype_cast(cols, dtypes):
    cols_out = []
    for col in cols:
        if col is None or col.dtype.type in dtypes:
            cols_out.append(col)
        else:
            cols_out.append(col.astype(dtypes[0]))
    return cols_out


class Direction(enum.Enum):
    ALL = 0
    IN = 1
    OUT = 2


def view_adj_list(input_graph):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
    if input_graph.adjlist is None:
        if input_graph.edgelist is None:
            raise ValueError('Graph is Empty')

        [src, dst] = datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        weights = None
        if input_graph.edgelist.weights:
            [weights] = datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])

        return coo2csr(src, dst, weights)


def view_transposed_adj_list(input_graph):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
    if input_graph.transposedadjlist is None:
        if input_graph.edgelist is None:
            if input_graph.adjlist is None:
                raise ValueError('Graph is Empty')
            else:
                input_graph.view_edge_list()

        [src, dst] = datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        weights = None
        if input_graph.edgelist.weights:
            [weights] = datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])

        return coo2csr(dst, src, weights)


def view_edge_list(input_graph):

    if input_graph.adjlist is None:
        raise RuntimeError('Graph is Empty')

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


def _degree_coo(edgelist_df, src_name, dst_name, direction=Direction.ALL, num_verts=None, sID=None):
    #
    #  Computing the degree of the input graph from COO
    #
    cdef DegreeDirection dir

    src = edgelist_df[src_name]
    dst = edgelist_df[dst_name]

    if direction == Direction.ALL:
        dir = DIRECTION_IN_PLUS_OUT
    elif direction == Direction.IN:
        dir = DIRECTION_IN
    elif direction == Direction.OUT:
        dir = DIRECTION_OUT
    else:
        raise ValueError("direction should be 0, 1 or 2")

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
        handle = Comms.get_handle(sID)
        handle_size_t = <size_t>handle.getHandle()
        graph.set_handle(<handle_t*>handle_size_t)

    graph.degree(<int*> c_degree, dir)
    graph.get_vertex_identifiers(<int*>c_vertex)

    return vertex_col, degree_col


def _degree_csr(offsets, indices, direction=Direction.ALL):
    cdef DegreeDirection dir

    if direction == Direction.ALL:
        dir = DIRECTION_IN_PLUS_OUT
    elif direction == Direction.IN:
        dir = DIRECTION_IN
    elif direction == Direction.OUT:
        dir = DIRECTION_OUT
    else:
        raise ValueError("direction should be 0, 1 or 2")

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


def _mg_degree(input_graph, direction=Direction.ALL):
    if input_graph.edgelist is None:
        input_graph.compute_renumber_edge_list(transposed=False)
    # The edge list renumbering step gives the columns that were renumbered
    # potentially new unique names.
    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    input_ddf = input_graph.edgelist.edgelist_df
    # Get the total number of vertices by summing each partition's number of vertices
    num_verts = input_graph.renumber_map.implementation.ddf.\
        map_partitions(len).compute().sum()
    data = DistributedDataHandler.create(data=input_ddf)
    comms = Comms.get_comms()
    client = default_client()
    data.calculate_parts_to_sizes(comms)
    if direction==Direction.IN:
        degree_ddf = [client.submit(_degree_coo,
                                    wf[1][0],
                                    src_col_name,
                                    dst_col_name,
                                    Direction.IN,
                                    num_verts,
                                    comms.sessionId,
                                    workers=[wf[0]])
                      for idx, wf in enumerate(data.worker_to_parts.items())]
    if direction==Direction.OUT:
        degree_ddf = [client.submit(_degree_coo,
                                    wf[1][0],
                                    dst_col_name,
                                    src_col_name,
                                    Direction.IN,
                                    num_verts,
                                    comms.sessionId,
                                    workers=[wf[0]])
                      for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(degree_ddf)
    return degree_ddf[0].result()


def _degree(input_graph, direction=Direction.ALL):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
    transpose_direction = { Direction.ALL: Direction.ALL,
                            Direction.IN: Direction.OUT,
                            Direction.OUT: Direction.IN }

    if input_graph.adjlist is not None:
        return _degree_csr(input_graph.adjlist.offsets,
                           input_graph.adjlist.indices,
                           direction)

    if input_graph.transposedadjlist is not None:
        return _degree_csr(input_graph.transposedadjlist.offsets,
                           input_graph.transposedadjlist.indices,
                           transpose_direction[direction])

    if input_graph.edgelist is not None:
        return _degree_coo(input_graph.edgelist.edgelist_df,
                           'src', 'dst', direction)

    raise ValueError("input_graph not COO, CSR or CSC")


def _degrees(input_graph):
    verts, indegrees = _degree(input_graph, Direction.IN)
    verts, outdegrees = _degree(input_graph, Direction.OUT)

    return verts, indegrees, outdegrees


# FIXME: this generates a cython warning about overriding a cdef method of the
# same name.
def get_two_hop_neighbors(input_graph):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
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
