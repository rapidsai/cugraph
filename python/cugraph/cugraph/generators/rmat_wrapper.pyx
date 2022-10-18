# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.utility cimport move, pair
from cython.operator cimport dereference as deref
import numpy as np

from rmm._lib.device_buffer cimport device_buffer
import cudf

from pylibraft.common.handle cimport handle_t
from cugraph.structure.graph_utilities cimport graph_generator_t
from cugraph.generators.rmat cimport (call_generate_rmat_edgelist,
                                      call_generate_rmat_edgelists,
                                      generator_distribution_t,
                                      UNIFORM,
                                      POWER_LAW,
                                      )
from cugraph.structure.graph_primtypes cimport move_device_buffer_to_column


def generate_rmat_edgelist(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    handle=None
):

    vertex_t = np.dtype("int32")
    if (2**scale) > (2**31 - 1):
        vertex_t = np.dtype("int64")

    cdef unique_ptr[handle_t] handle_ptr
    cdef size_t handle_size_t

    if handle is None:
        handle_ptr.reset(new handle_t())
        handle_ = handle_ptr.get()
    else:
        handle_size_t = <size_t>handle.getHandle()
        handle_ = <handle_t*>handle_size_t

    cdef unique_ptr[graph_generator_t] gg_ret_ptr

    if (vertex_t==np.dtype("int32")):
        gg_ret_ptr = move(call_generate_rmat_edgelist[int]( deref(handle_),
                                                    scale,
                                                    num_edges,
                                                    a,
                                                    b,
                                                    c,
                                                    seed,
                                                    clip_and_flip,
                                                    scramble_vertex_ids))
    else: # (vertex_t == np.dtype("int64"))
        gg_ret_ptr = move(call_generate_rmat_edgelist[long]( deref(handle_),
                                                    scale,
                                                    num_edges,
                                                    a,
                                                    b,
                                                    c,
                                                    seed,
                                                    clip_and_flip,
                                                    scramble_vertex_ids))

    gg_ret = move(gg_ret_ptr.get()[0])

    set_source = move_device_buffer_to_column(move(gg_ret.d_source), vertex_t)
    set_destination = move_device_buffer_to_column(move(gg_ret.d_destination), vertex_t)

    df = cudf.DataFrame()
    df['src'] = set_source
    df['dst'] = set_destination

    return df


def generate_rmat_edgelists(
    n_edgelists,
    min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids
    ):

    vertex_t = np.dtype("int32")
    if (2**max_scale) > (2**31 - 1):
        vertex_t = np.dtype("int64")

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    cdef generator_distribution_t s_distribution
    cdef generator_distribution_t e_distribution
    if size_distribution == 0:
        s_distribution= POWER_LAW
    else :
        s_distribution= UNIFORM
    if edge_distribution == 0:
        e_distribution= POWER_LAW
    else :
        e_distribution= UNIFORM

    cdef vector[pair[unique_ptr[device_buffer], unique_ptr[device_buffer]]] gg_ret_ptr

    if (vertex_t==np.dtype("int32")):
         gg_ret_ptr = move(call_generate_rmat_edgelists[int]( deref(handle_),
                                                    n_edgelists,
                                                    min_scale,
                                                    max_scale,
                                                    edge_factor,
                                                    <generator_distribution_t>s_distribution,
                                                    <generator_distribution_t>e_distribution,
                                                    seed,
                                                    clip_and_flip,
                                                    scramble_vertex_ids))
    else: # (vertex_t == np.dtype("int64"))
         gg_ret_ptr = move(call_generate_rmat_edgelists[long]( deref(handle_),
                                                    n_edgelists,
                                                    min_scale,
                                                    max_scale,
                                                    edge_factor,
                                                    <generator_distribution_t>s_distribution,
                                                    <generator_distribution_t>e_distribution,
                                                    seed,
                                                    clip_and_flip,
                                                    scramble_vertex_ids))
    list_df = []

    for i in range(n_edgelists):
        set_source = move_device_buffer_to_column(move(gg_ret_ptr[i].first), vertex_t)
        set_destination = move_device_buffer_to_column(move(gg_ret_ptr[i].second), vertex_t)

        df = cudf.DataFrame()
        df['src'] = set_source
        df['dst'] = set_destination

        list_df.append(df)

    #Return a list of dataframes
    return list_df
