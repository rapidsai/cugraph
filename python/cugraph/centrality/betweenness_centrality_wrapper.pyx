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

from cugraph.centrality.betweenness_centrality cimport betweenness_centrality as c_betweenness_centrality
from cugraph.centrality.betweenness_centrality cimport handle_t
from cugraph.structure import graph_new_wrapper
from cugraph.structure.graph import DiGraph
from cugraph.structure.graph_new cimport *
from cugraph.utilities.unrenumber import unrenumber
from libc.stdint cimport uintptr_t
from libcpp cimport bool
import cudf
import numpy as np
import numpy.ctypeslib as ctypeslib

import dask_cudf
import dask_cuda
import cugraph.raft
from cugraph.raft.dask.common.comms import (Comms, worker_state)
from cugraph.raft.dask.common.utils import default_client
import dask.distributed


def get_output_df(number_of_vertices, result_dtype):
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(number_of_vertices, dtype=np.int32))
    df['betweenness_centrality'] = cudf.Series(np.zeros(number_of_vertices,
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


# TODO(xcadet) There might be an issue with weights, should be duplicated at
# c++ level
def run_work(input_graph, normalized, endpoints, sources, weights,
             result_dtype, session_id):
    result = None
    # 1. Get session information
    session_state = worker_state(session_id)
    number_of_workers = session_state["nworkers"]
    worker_idx = session_state["wid"]

    # 2. Get handle
    handle = session_state['handle']

    # 3. Get Batch # TODO(xcadet): This maybe be directly handled deeper
    batch = get_batch(sources, number_of_workers, worker_idx)

    # 4. Determine worker type
    is_organizer = (worker_idx == 0) # TODO(xcadet) Refactor for clarity
    total_number_of_sources = len(sources)

    # 5. Dispatch to proper type
    if is_organizer:
        result = run_organizer_work(handle, input_graph, normalized,
                                    endpoints, batch, weights,
                                    total_number_of_sources, result_dtype)
    else:
        result = run_regular_work(handle, normalized, endpoints,
                                  batch, weights,
                                  total_number_of_sources, result_dtype)

    return result


def run_organizer_work(handle, input_graph, normalized, endpoints, batch,
                       weights,
                       total_number_of_sources, result_dtype):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_graph = <uintptr_t> NULL
    cdef uintptr_t c_identifier = <uintptr_t> NULL
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL
    cdef uintptr_t c_batch = <uintptr_t> NULL
    # TODO(xcadet) look into a way to merge them
    cdef GraphCSRViewDouble graph_double
    cdef GraphCSRViewFloat graph_float

    result_size = input_graph.number_of_vertices()
    result_df = get_output_df(result_size, result_dtype)
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

    c_identifier = result_df['vertex'].__cuda_array_interface__['data'][0]
    c_betweenness = result_df['betweenness_centrality'].__cuda_array_interface__['data'][0]
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]
    c_batch = batch.__array_interface__['data'][0]
    c_handle = <uintptr_t>handle.getHandle()

    run_c_betweenness_centrality(c_handle,
                                 c_graph,
                                 c_betweenness,
                                 normalized,
                                 endpoints,
                                 c_weights,
                                 number_of_sources_in_batch,
                                 c_batch,
                                 total_number_of_sources,
                                 result_dtype)
    if result_dtype == np.float64:
        graph_double.get_vertex_identifiers(<int*> c_identifier)
    elif result_dtype == np.float32:
        graph_float.get_vertex_identifiers(<int*> c_identifier)
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")

    return result_df


def run_regular_work(handle, normalized, endpoints, batch, weights,
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

    run_c_betweenness_centrality(c_handle,
                                 c_graph,
                                 c_betweenness,
                                 normalized,
                                 endpoints,
                                 c_weights,
                                 number_of_sources_in_batch,
                                 c_batch,
                                 total_number_of_sources,
                                 result_dtype)
    return None


cdef void run_c_betweenness_centrality(uintptr_t c_handle,
                                       uintptr_t c_graph,
                                       uintptr_t c_betweenness,
                                       bool normalized,
                                       bool endpoints,
                                       uintptr_t c_weights,
                                       int number_of_sources_in_batch,
                                       uintptr_t c_batch,
                                       int total_number_of_sources,
                                       result_dtype):
    if result_dtype == np.float64:
        c_betweenness_centrality[int, int, double, double]((<handle_t *> c_handle)[0],
                                                           <GraphCSRView[int, int, double] *> c_graph,
                                                           <double *> c_betweenness,
                                                           normalized,
                                                           endpoints,
                                                           <double *> c_weights,
                                                           number_of_sources_in_batch,
                                                           <int *> c_batch,
                                                           total_number_of_sources)
    elif result_dtype == np.float32:
        c_betweenness_centrality[int, int, float, float]((<handle_t *> c_handle)[0],
                                                         <GraphCSRView[int, int, float] *> c_graph,
                                                         <float *> c_betweenness,
                                                         normalized,
                                                         endpoints,
                                                         <float *> c_weights,
                                                         number_of_sources_in_batch,
                                                         <int *> c_batch,
                                                         total_number_of_sources)
    else:
        raise ValueError("result_dtype can only be np.float64 or np.float32")


def opg_get_client():
    try:
        client = default_client()
    except ValueError:
        client = None

    return client


def opg_get_comms_using_client(client):
    comms = None

    if client is not None:
        comms = Comms(client=client)

    return comms


def get_numpy_vertices(vertices, graph_number_of_vertices):
    if vertices is not None:
        numpy_vertices =  np.array(vertices, dtype=np.int32)
    else:
        numpy_vertices = np.arange(graph_number_of_vertices, dtype=np.int32)

    return numpy_vertices


class CommsContext:
    def __init__(self, comms):
        self._comms = comms

    def __enter__(self):
        self._comms.init()

    def __exit__(self, type, value, traceback):
        self._comms.destroy()


def betweenness_centrality(input_graph, normalized, endpoints, weights,
                           vertices, result_dtype):
    """
    Call betweenness centrality
    """
    cdef GraphCSRViewFloat graph_float
    cdef GraphCSRViewDouble graph_double
    cdef uintptr_t c_identifier = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL
    cdef uintptr_t c_vertices = <uintptr_t> NULL
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef handle_t *c_handle

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

    numpy_vertices =  get_numpy_vertices(vertices, input_graph.number_of_vertices())
    c_vertices = numpy_vertices.__array_interface__['data'][0]
    k  = len(numpy_vertices)

    # TODO(xcadet) Check if there is a better way to do this
    # NOTE: The current implementation only has <int, int, float, float> and
    #       <int, int, double, double> as explicit template declaration
    #       The current BFS requires the GraphCSR to be declared
    #       as <int, int, float> or <int, int double> even if weights is null
    client = opg_get_client()
    comms = opg_get_comms_using_client(client)
    if result_dtype == np.float32:
        if comms is not None:
            with CommsContext(comms):
                main_worker = comms.worker_addresses[0]
                future_pointer_to_input_graph = client.scatter(input_graph, workers=[main_worker])
                futures = [client.submit(run_work,
                                         future_pointer_to_input_graph if worker_idx == 0 else input_graph.number_of_vertices(),
                                         normalized,
                                         endpoints,
                                         numpy_vertices,
                                         weights,
                                         result_dtype, comms.sessionId,
                                         workers=[worker_address]) for worker_idx, worker_address in enumerate(comms.worker_addresses)]
                dask.distributed.wait(futures)
                df = futures[0].result()
        else:
            df = get_output_df(input_graph, result_dtype)

            c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
            c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0]

            graph_float = get_graph_view[GraphCSRViewFloat](input_graph, False)
            graph_float.prop.directed = type(input_graph) is DiGraph
            handle = cugraph.raft.common.handle.Handle() #  if handle is None else handle
            c_handle = <handle_t*><size_t> handle.getHandle()
            total_number_of_sources_used = len(numpy_vertices)
            c_betweenness_centrality[int, int, float, float](c_handle[0],
                                                            &graph_float,
                                                            <float*> c_betweenness,
                                                            normalized, endpoints,
                                                            <float*> c_weights,
                                                            k,
                                                            <int*> c_vertices,
                                                            total_number_of_sources_used)
            graph_float.get_vertex_identifiers(<int*> c_identifier)

    elif result_dtype == np.float64:
        if comms is not None:
            with CommsContext(comms):
                main_worker = comms.worker_addresses[0]
                future_pointer_to_input_graph = client.scatter(input_graph, workers=[main_worker])
                futures = [client.submit(run_work,
                                         future_pointer_to_input_graph if worker_idx == 0 else input_graph.number_of_vertices(),
                                         normalized,
                                         endpoints,
                                         numpy_vertices,
                                         weights,
                                         result_dtype, comms.sessionId,
                                         workers=[worker_address]) for worker_idx, worker_address in enumerate(comms.worker_addresses)]
                dask.distributed.wait(futures)
                df = futures[0].result()
        else:
            df = get_output_df(input_graph, result_dtype)

            c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
            c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0]

            graph_double = get_graph_view[GraphCSRViewDouble](input_graph, False)
            graph_double.prop.directed = type(input_graph) is DiGraph
            handle = cugraph.raft.common.handle.Handle() #  if handle is None else handle
            c_handle = <handle_t*><size_t> handle.getHandle()
            total_number_of_sources_used = len(numpy_vertices)
            c_betweenness_centrality[int, int, double, double](c_handle[0],
                                                               &graph_double,
                                                               <double*> c_betweenness,
                                                               normalized, endpoints,
                                                               <double*> c_weights,
                                                               k,
                                                               <int*> c_vertices,
                                                               total_number_of_sources_used)
            graph_double.get_vertex_identifiers(<int*> c_identifier)

    else:
        raise TypeError("result type for betweenness centrality can only be "
                        "float or double")

    # For large graph unrenumbering produces a dataframe organized
    #       in buckets, i.e, if they are 3 buckets
    # 0
    # 8191
    # 16382
    # 1
    # 8192 ...
    # Instead of having  the sources in ascending order
    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')

    return df
