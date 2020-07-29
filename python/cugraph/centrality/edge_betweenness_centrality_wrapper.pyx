# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cugraph.centrality.betweenness_centrality cimport edge_betweenness_centrality as c_edge_betweenness_centrality
from cugraph.structure import graph_new_wrapper
from cugraph.structure.graph import DiGraph, Graph
from cugraph.structure.graph_new cimport *
from cugraph.utilities.unrenumber import unrenumber
from libc.stdint cimport uintptr_t
from libcpp cimport bool
import cudf
import numpy as np
import numpy.ctypeslib as ctypeslib

import cugraph.raft
from cugraph.dask.common.mg_utils import (mg_get_client,
                                          is_worker_organizer)
import cugraph.comms.comms as Comms
from cugraph.raft.dask.common.comms import worker_state
import dask.distributed
from cugraph.structure.utils_wrapper import coo2csr


def get_sg_output_df(input_graph, result_dtype):
    number_of_edges = input_graph.number_of_edges(directed_edges=True)
    df = cudf.DataFrame()
    df['src'] = cudf.Series(np.zeros(number_of_edges, dtype=np.int32))
    df['dst'] = input_graph.adjlist.indices.copy()
    df['betweenness_centrality'] = cudf.Series(np.zeros(number_of_edges,
                                               dtype=result_dtype))
    return df

def get_mg_output_df(src, indices, result_dtype):
    number_of_edges = len(src)
    df = cudf.DataFrame()
    df['src'] = src.copy()
    df['dst'] = indices.copy()
    df['betweenness_centrality'] = cudf.Series(np.zeros(number_of_edges,
                                               dtype=result_dtype))
    return df


def get_batch(sources, number_of_workers, current_worker):
    batch_size = len(sources) // number_of_workers
    begin =  current_worker * batch_size
    end = (current_worker + 1) * batch_size
    if current_worker == (number_of_workers - 1):
        end = len(sources)
    batch = sources[begin:end]
    return batch


def run_work(input_graph, normalized, weights, sources,
             result_dtype, session_id):
    result = None
    # 1. Get session information
    session_state = worker_state(session_id)
    number_of_workers = session_state["nworkers"]
    worker_idx = session_state["wid"]

    # 2. Get handle
    handle = session_state['handle']

    # 3. Get Batch
    batch = get_batch(sources, number_of_workers, worker_idx)

    # 4. Determine worker type
    is_organizer = is_worker_organizer(worker_idx)
    total_number_of_sources = len(sources)

    # 5. Dispatch to proper type
    if is_organizer:
        result = run_organizer_work(handle, input_graph, normalized, weights,
                                    batch, total_number_of_sources,
                                    result_dtype)
    else:
        result = run_regular_work(handle, normalized, weights, batch,
                                  total_number_of_sources, result_dtype)

    return result


def run_organizer_work(handle, input_data, normalized, weights, batch,
                       total_number_of_sources, result_dtype):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_graph = <uintptr_t> NULL
    cdef uintptr_t c_src_identifier = <uintptr_t> NULL
    cdef uintptr_t c_dst_identifier = <uintptr_t> NULL
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL
    cdef uintptr_t c_batch = <uintptr_t> NULL

    cdef uintptr_t c_offsets = <uintptr_t> NULL
    cdef uintptr_t c_indices = <uintptr_t> NULL
    cdef uintptr_t c_graph_weights = <uintptr_t> NULL

    cdef GraphCSRViewDouble graph_double
    cdef GraphCSRViewFloat graph_float

    _data, local_data, is_directed =  input_data

    data = _data[0]
    src =  data['src']
    dst =  data['dst']
    src, dst = graph_new_wrapper.datatype_cast([src, dst], [np.int32])
    offsets, indices, graph_weights = coo2csr(src, dst, None)
    if graph_weights:
        c_graph_weights = graph_weights.__cuda_array_interface__['data'][0]
    c_offsets = offsets.__cuda_array_interface__['data'][0]
    c_indices = indices.__cuda_array_interface__['data'][0]

    number_of_vertices = local_data['verts'].sum()
    number_of_edges = local_data['edges'].sum()

    result_df = get_mg_output_df(src, indices, result_dtype)
    number_of_sources_in_batch = len(batch)
    if result_dtype == np.float64:
        graph_double = GraphCSRView[int, int, double](<int*> c_offsets,
                                                      <int*> c_indices,
                                                      <double*> c_graph_weights,
                                                      number_of_vertices,
                                                      number_of_edges)
        graph_double.prop.directed = is_directed
        c_graph = <uintptr_t>&graph_double
    elif result_dtype == np.float32:
        graph_float = GraphCSRView[int, int, float](<int*>c_offsets,
                                                    <int*>c_indices,
                                                    <float*>c_graph_weights,
                                                    number_of_vertices,
                                                    number_of_edges)
        graph_float.prop.directed = is_directed
        c_graph = <uintptr_t>&graph_float
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

    c_src_identifier = result_df['src'].__cuda_array_interface__['data'][0]
    c_dst_identifier = result_df['dst'].__cuda_array_interface__['data'][0]
    c_betweenness = result_df['betweenness_centrality'].__cuda_array_interface__['data'][0]
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]
    c_batch = batch.__array_interface__['data'][0]
    c_handle = <uintptr_t>handle.getHandle()

    run_c_edge_betweenness_centrality(c_handle,
                                      c_graph,
                                      c_betweenness,
                                      normalized,
                                      c_weights,
                                      number_of_sources_in_batch,
                                      c_batch,
                                      total_number_of_sources,
                                      result_dtype)
    if result_dtype == np.float64:
        graph_double.get_source_indices(<int*>c_src_identifier)
    elif result_dtype == np.float32:
        graph_float.get_source_indices(<int*>c_src_identifier)
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

    return result_df


def run_regular_work(handle, normalized, weights, batch,
                     total_number_of_sources, result_dtype):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_graph = <uintptr_t> NULL
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_batch = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL

    number_of_sources_in_batch = len(batch)

    c_batch = batch.__array_interface__['data'][0]
    c_handle = <uintptr_t>handle.getHandle()
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

    run_c_edge_betweenness_centrality(c_handle,
                                      c_graph,
                                      c_betweenness,
                                      normalized,
                                      c_weights,
                                      number_of_sources_in_batch,
                                      c_batch,
                                      total_number_of_sources,
                                      result_dtype)
    return None


def run_sg_work(handle, input_graph, normalized,
                weights, batch,
                total_number_of_sources, result_dtype):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_graph = <uintptr_t> NULL
    cdef uintptr_t c_src_identifier = <uintptr_t> NULL
    cdef uintptr_t c_dst_identifier = <uintptr_t> NULL
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL
    cdef uintptr_t c_batch = <uintptr_t> NULL

    cdef GraphCSRViewDouble graph_double
    cdef GraphCSRViewFloat graph_float

    result_df = get_sg_output_df(input_graph, result_dtype)
    number_of_sources_in_batch = len(batch)
    if result_dtype == np.float64:
        graph_double = get_graph_view[GraphCSRViewDouble](input_graph, False)
        graph_double.prop.directed = type(input_graph) is DiGraph
        c_graph = <uintptr_t>&graph_double
    elif result_dtype == np.float32:
        graph_float = get_graph_view[GraphCSRViewFloat](input_graph, False)
        graph_float.prop.directed = type(input_graph) is DiGraph
        c_graph = <uintptr_t>&graph_float
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

    c_src_identifier = result_df['src'].__cuda_array_interface__['data'][0]
    c_dst_identifier = result_df['dst'].__cuda_array_interface__['data'][0]
    c_betweenness = result_df['betweenness_centrality'].__cuda_array_interface__['data'][0]
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]
    c_batch = batch.__array_interface__['data'][0]
    c_handle = <uintptr_t>handle.getHandle()

    run_c_edge_betweenness_centrality(c_handle,
                                      c_graph,
                                      c_betweenness,
                                      normalized,
                                      c_weights,
                                      number_of_sources_in_batch,
                                      c_batch,
                                      total_number_of_sources,
                                      result_dtype)
    if result_dtype == np.float64:
        graph_double.get_source_indices(<int*>c_src_identifier)
    elif result_dtype == np.float32:
        graph_float.get_source_indices(<int*>c_src_identifier)
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

    return result_df


cdef void run_c_edge_betweenness_centrality(uintptr_t c_handle,
                                            uintptr_t c_graph,
                                            uintptr_t c_betweenness,
                                            bool normalized,
                                            uintptr_t c_weights,
                                            int number_of_sources_in_batch,
                                            uintptr_t c_batch,
                                            int total_number_of_sources,
                                            result_dtype):
    if result_dtype == np.float64:
        c_edge_betweenness_centrality[int, int, double, double]((<handle_t *> c_handle)[0],
                                                                <GraphCSRView[int, int, double] *> c_graph,
                                                                <double *> c_betweenness,
                                                                normalized,
                                                                <double *> c_weights,
                                                                number_of_sources_in_batch,
                                                                <int *> c_batch,
                                                                total_number_of_sources)
    elif result_dtype == np.float32:
        c_edge_betweenness_centrality[int, int, float, float]((<handle_t *> c_handle)[0],
                                                              <GraphCSRView[int, int, float] *> c_graph,
                                                              <float *> c_betweenness,
                                                              normalized,
                                                              <float *> c_weights,
                                                              number_of_sources_in_batch,
                                                              <int *> c_batch,
                                                              total_number_of_sources)
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

def mg_batch_edge_betweenness_centrality(client, comms, input_graph,
                                         normalized,
                                         weights, vertices, result_dtype):
    df = None
    data  = cugraph.dask.common.input_utils.get_mg_batch_data(input_graph)
    for placeholder, worker in enumerate(client.has_what().keys()):
        if worker not in  data.worker_to_parts:
            data.worker_to_parts[worker] = [[placeholder], None]
    work_futures =  [client.submit(run_work,
                                   (wf[1], data.local_data,
                                    type(input_graph) is cugraph.DiGraph),
                                   normalized,
                                   weights,
                                   vertices,
                                   result_dtype,
                                   comms.sessionId,
                                   workers=[wf[0]]) for
                     idx, wf in enumerate(data.worker_to_parts.items())]
    dask.distributed.wait(work_futures)
    df = work_futures[0].result()
    return df


def sg_edge_betweenness_centrality(input_graph, normalized, weights,
                                   vertices, result_dtype):
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    total_number_of_sources = len(vertices)
    handle = cugraph.raft.common.handle.Handle()
    df = run_sg_work(handle, input_graph, normalized, weights,
                     vertices, total_number_of_sources, result_dtype)
    return df



def edge_betweenness_centrality(input_graph, normalized, weights,
                                vertices, result_dtype):
    """
    Call betweenness centrality
    """
    df = None

    client = mg_get_client()
    comms = Comms.get_comms()
    if comms:
        assert input_graph.replicatable == True, "To run Batch Analytics on " \
            "Multi GPU, the graph needs to be "     \
            "located on a single GPU"
        df = mg_batch_edge_betweenness_centrality(client, comms, input_graph, normalized,
                                                  weights, vertices,
                                                  result_dtype)
    else:
        df = sg_edge_betweenness_centrality(input_graph, normalized,
                                            weights, vertices, result_dtype)

    # Same as Betweenness Centrality unrenumber resuls might be organized
    # in buckets
    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'src')
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'dst')

    if type(input_graph) is Graph:
        lower_triangle = df['src'] >= df['dst']
        df[["src", "dst"]][lower_triangle] = df[["dst", "src"]][lower_triangle]
        df = df.groupby(by=["src", "dst"]).sum().reset_index()

    return df

