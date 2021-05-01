# Copyright (c) 2021, NVIDIA CORPORATION.
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

from libcpp cimport bool
from libc.stdint cimport uintptr_t
import numpy as np
import numpy.ctypeslib as ctypeslib
from cython.operator cimport dereference as deref

import rmm
from rmm._lib.device_buffer cimport DeviceBuffer
import cudf
from cudf.core.buffer import Buffer

from cugraph.structure.graph_utilities cimport *  #This line should be determined as well
from cugraph.generators.rmat cimport *
from libcpp.utility cimport move  # This must be imported after graph_utilities
                                  # since graph_utilities also defines move


def generate_rmat_edgelist(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids
):

    vertex_t = np.dtype("int32")
    if (2**scale) > (2**31 - 1):
        vertex_t = np.dtype("int64")

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()


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

    gg_ret= move(gg_ret_ptr.get()[0])
    source_set = DeviceBuffer.c_from_unique_ptr(move(gg_ret.d_source))
    destination_set = DeviceBuffer.c_from_unique_ptr(move(gg_ret.d_destination))
    source_set = Buffer(source_set)
    destination_set = Buffer(destination_set)

    set_source = cudf.Series(data=source_set, dtype=vertex_t)
    set_destination = cudf.Series(data=destination_set, dtype=vertex_t)

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
    if max_scale > (2**31 - 1):
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
    cdef unique_ptr[graph_generator_t*] gg_ret_ptr

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
    list_df = []
    gg_ret= move(gg_ret_ptr.get()[0])
    for i in range(n_edgelists):
        source_set = DeviceBuffer.c_from_unique_ptr(move(gg_ret[i].d_source))
        destination_set = DeviceBuffer.c_from_unique_ptr(move(gg_ret[i].d_destination))
        source_set = Buffer(source_set)
        destination_set = Buffer(destination_set)

        set_source = cudf.Series(data=source_set, dtype=vertex_t)
        set_destination = cudf.Series(data=destination_set, dtype=vertex_t)

        df = cudf.DataFrame()
        df['src'] = set_source
        df['dst'] = set_destination

        list_df.append(df)

    #Return a list of dataframes
    return list_df
