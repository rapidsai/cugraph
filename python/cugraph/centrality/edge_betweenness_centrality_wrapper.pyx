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
from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.graph import DiGraph, Graph
from cugraph.structure.graph_primtypes cimport *
from libc.stdint cimport uintptr_t
from libcpp cimport bool
import cudf
import numpy as np
import numpy.ctypeslib as ctypeslib

from cugraph.dask.common.mg_utils import get_client
import cugraph.comms.comms as Comms
import dask.distributed


def get_output_df(indices, result_dtype):
    number_of_edges = len(indices)
    df = cudf.DataFrame()
    df['src'] = cudf.Series(np.zeros(number_of_edges, dtype=np.int32))
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


def run_mg_work(input_data, normalized, weights, sources,
             result_dtype, session_id):
    result = None

    number_of_workers = Comms.get_n_workers(session_id)
    worker_idx = Comms.get_worker_id(session_id)
    handle = Comms.get_handle(session_id)

    batch = get_batch(sources, number_of_workers, worker_idx)

    result = run_internal_work(handle, input_data, normalized, weights,
                               batch, result_dtype)
    return result


def run_internal_work(handle, input_data, normalized, weights, batch,
                      result_dtype):
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

    (offsets, indices, graph_weights), is_directed =  input_data

    if graph_weights is not None:
        c_graph_weights = graph_weights.__cuda_array_interface__['data'][0]
    c_offsets = offsets.__cuda_array_interface__['data'][0]
    c_indices = indices.__cuda_array_interface__['data'][0]

    number_of_vertices = len(offsets) - 1
    number_of_edges = len(indices)

    result_df = get_output_df(indices, result_dtype)
    c_src_identifier = result_df['src'].__cuda_array_interface__['data'][0]
    c_dst_identifier = result_df['dst'].__cuda_array_interface__['data'][0]
    c_betweenness = result_df['betweenness_centrality'].__cuda_array_interface__['data'][0]

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
                                      result_dtype)
    return result_df


cdef void run_c_edge_betweenness_centrality(uintptr_t c_handle,
                                            uintptr_t c_graph,
                                            uintptr_t c_betweenness,
                                            bool normalized,
                                            uintptr_t c_weights,
                                            int number_of_sources_in_batch,
                                            uintptr_t c_batch,
                                            result_dtype):
    if result_dtype == np.float64:
        c_edge_betweenness_centrality[int, int, double, double]((<handle_t *> c_handle)[0],
                                                                (<GraphCSRView[int, int, double] *> c_graph)[0],
                                                                <double *> c_betweenness,
                                                                normalized,
                                                                <double *> c_weights,
                                                                number_of_sources_in_batch,
                                                                <int *> c_batch)
    elif result_dtype == np.float32:
        c_edge_betweenness_centrality[int, int, float, float]((<handle_t *> c_handle)[0],
                                                              (<GraphCSRView[int, int, float] *> c_graph)[0],
                                                              <float *> c_betweenness,
                                                              normalized,
                                                              <float *> c_weights,
                                                              number_of_sources_in_batch,
                                                              <int *> c_batch)
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

def batch_edge_betweenness_centrality(input_graph,
                                         normalized,
                                         weights, vertices, result_dtype):
    client = get_client()
    comms = Comms.get_comms()
    replicated_adjlists = input_graph.batch_adjlists
    work_futures =  [client.submit(run_mg_work,
                                   (data, type(input_graph)
                                   is DiGraph),
                                   normalized,
                                   weights,
                                   vertices,
                                   result_dtype,
                                   comms.sessionId,
                                   workers=[worker]) for
                    (worker, data) in replicated_adjlists.items()]
    dask.distributed.wait(work_futures)
    df = work_futures[0].result()
    return df


def sg_edge_betweenness_centrality(input_graph, normalized, weights,
                                   vertices, result_dtype):
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    handle = Comms.get_default_handle()
    adjlist = input_graph.adjlist
    input_data = ((adjlist.offsets, adjlist.indices, adjlist.weights),
                  type(input_graph) is DiGraph)
    df = run_internal_work(handle, input_data, normalized, weights,
                           vertices, result_dtype)
    return df


def edge_betweenness_centrality(input_graph, normalized, weights,
                                vertices, result_dtype):
    """
    Call betweenness centrality
    """
    cdef GraphCSRViewDouble graph_double
    cdef GraphCSRViewFloat graph_float


    df = None

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    if Comms.is_initialized() and input_graph.batch_enabled == True:
        df = batch_edge_betweenness_centrality(input_graph, normalized,
                                                  weights, vertices,
                                                  result_dtype)
    else:
        df = sg_edge_betweenness_centrality(input_graph, normalized,
                                            weights, vertices, result_dtype)

    if result_dtype == np.float64:
        graph_double = get_graph_view[GraphCSRViewDouble](input_graph)
        graph_double.get_source_indices(<int*>(<uintptr_t>df['src'].__cuda_array_interface__['data'][0]))
    elif result_dtype == np.float32:
        graph_float = get_graph_view[GraphCSRViewFloat](input_graph)
        graph_float.get_source_indices(<int*>(<uintptr_t>df['src'].__cuda_array_interface__['data'][0]))

    return df
