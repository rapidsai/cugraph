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
from cugraph.raft.dask.common.comms import Comms
from dask.distributed import wait, default_client
from cugraph.raft.dask.common.comms import worker_state
#from cugraph.dask.common.input_utils import DistributedDataHandler

def prepare_client():
    client = default_client()
    return client

def prepare_comms(client):
     comms = Comms(comms_p2p=False)
     comms.init()
     return comms

def prepare_batch_opg():
    client = prepare_client()
    prepare_comms(client)
    # client = default_client()
    # futures = [client.submit(task ...]
    # wait(degree_ddf)


def get_output_df(input_graph, result_dtype):
    number_of_vertices = input_graph.number_of_vertices()
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


def run_work(input_graph, normalized, endpoints, vertices, result_dtype, session_id):
    df = get_output_df(input_graph, result_dtype)
    session_state = worker_state(session_id)
    number_of_workers = session_state['nworkers']
    worker_id = session_state['wid']

    cdef GraphCSRViewFloat graph_float = get_graph_view[GraphCSRViewFloat](input_graph, False)
    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_batch = <uintptr_t> NULL

    batch = get_batch(vertices, number_of_workers, worker_id)
    number_of_sources_in_batch = len(batch)
    print("[DBG] Worker {}/{} has to process batch({}) = {}".format(worker_id + 1, number_of_workers, number_of_sources_in_batch, batch))
    c_batch = batch.__array_interface__['data'][0]

    handle = session_state['handle']
    c_handle = <handle_t*><size_t> handle.getHandle()
    c_betweenness_centrality[int, int, float, float](c_handle[0],
                                                     graph_float,
                                                     <float*> c_betweenness,
                                                     normalized,
                                                     endpoints,
                                                     <float*> NULL,
                                                     number_of_sources_in_batch,
                                                     <int*> c_batch)
    graph_float.get_vertex_identifiers(<int*> c_identifier)
    if worker_id == 0:
        return df


def betweenness_centrality(input_graph, normalized, endpoints, weight, k,
                           vertices, result_dtype):
    """
    Call betweenness centrality
    """
    cdef GraphCSRViewFloat graph_float
    cdef GraphCSRViewDouble graph_double
    cdef uintptr_t c_identifier = <uintptr_t> NULL
    cdef uintptr_t c_betweenness = <uintptr_t> NULL
    cdef uintptr_t c_vertices = <uintptr_t> NULL
    cdef uintptr_t c_weight = <uintptr_t> NULL
    cdef handle_t *c_handle

    if not input_graph.adjlist:
        input_graph.view_adj_list()

    if weight is not None:
        c_weight = weight.__cuda_array_interface__['data'][0]

    if vertices is not None:
        # NOTE: Do not merge lines, c_vertices may end up pointing at the
        #       wrong place the length of vertices increase.
        np_verts =  np.array(vertices, dtype=np.int32)

    c_k = 0
    if k is not None:
        c_k = k
    # TODO(xcadet) Check if there is a better way to do this
    # TODO(xcadet) Find a way to execute without a Client
    # prepare_batch_opg()
    # NOTE: The current implementation only has <int, int, float, float> and
    #       <int, int, double, double> as explicit template declaration
    #       The current BFS requires the GraphCSR to be declared
    #       as <int, int, float> or <int, int double> even if weights is null
    if result_dtype == np.float32:
        client = default_client()
        comms = Comms(client=client, comms_p2p=False)
        comms.init()


        if comms is not None:
            df = get_output_df(input_graph, result_dtype)
            if vertices is None:
                vertices = np.arange(input_graph.number_of_vertices(), dtype=np.int32)
            futures = [client.submit(run_work, input_graph, normalized, endpoints, vertices, result_dtype, comms.sessionId, workers=[worker_id]) for worker_id in comms.worker_addresses]
            wait(futures)
            df = futures[0].result()
        else:
            df = get_output_df(input_graph, result_dtype)

            c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
            c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0]


            graph_float = get_graph_view[GraphCSRViewFloat](input_graph, False)
            graph_float.prop.directed = type(input_graph) is DiGraph
            handle = cugraph.raft.common.handle.Handle() #  if handle is None else handle
            c_handle = <handle_t*><size_t> handle.getHandle()
            c_betweenness_centrality[int, int, float, float](c_handle[0],
                                                            graph_float,
                                                            <float*> c_betweenness,
                                                            normalized, endpoints,
                                                            <float*> c_weight, c_k,
                                                            <int*> c_vertices)
            graph_float.get_vertex_identifiers(<int*> c_identifier)

    elif result_dtype == np.float64:
        df = get_output_df(input_graph, result_dtype)
        df = dask_cuda.persist(df)

        c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
        c_betweenness = df['betweenness_centrality'].__cuda_array_interface__['data'][0]


        graph_double = get_graph_view[GraphCSRViewDouble](input_graph, False)
        # FIXME: There might be a way to avoid manually setting the Graph property
        graph_double.prop.directed = type(input_graph) is DiGraph
        handle = cugraph.raft.common.handle.Handle() #  if handle is None else handle
        c_handle = <handle_t*><size_t> handle.getHandle()
        c_betweenness_centrality[int, int, double, double](c_handle[0],
                                                           graph_double,
                                                           <double*> c_betweenness,
                                                           normalized, endpoints,
                                                           <double*> c_weight, c_k,
                                                           <int*> c_vertices)
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
