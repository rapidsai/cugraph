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
from cugraph.utilities.graph_generator cimport call_generate_rmat_edgelist #To be determined
from cugraph.structure.graph_utilities cimport *  #This line should be determined as well
from libcpp cimport bool
from libcpp.utility cimport move
from libc.stdint cimport uintptr_t
import cudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib
from rmm._lib.device_buffer cimport DeviceBuffer
from cudf.core.buffer import Buffer
from cython.operator cimport dereference as deref
def graph_generator(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids
):

    #vertex_t = np.dtype("int32")
    edge_t = np.dtype("int32")
    if num_edges > (2**31 - 1):
        edge_t = np.dtype("int64")
    #cdef unique_ptr[random_walk_ret_t] rw_ret_ptr I do not need a pointer
    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    
    cdef unique_ptr[graph_generator_t] gg_ret_ptr 
    #cdef tuple gg_ret_ptr
    #"""
    move(call_generate_rmat_edgelist[int]( deref(handle_),
                                                   scale,
                                                   num_edges,
                                                   a,
                                                   b,
                                                   c,
                                                   seed,
                                                   clip_and_flip,
                                                   scramble_vertex_ids))
    
    #gg_ret_ptr = Buffer(gg_ret_ptr)
    gg_ret= move(gg_ret_ptr.get()[0])
    source_set = DeviceBuffer.c_from_unique_ptr(move(gg_ret.d_source))
    destination_set = DeviceBuffer.c_from_unique_ptr(move(gg_ret.d_destination))
    source_set = Buffer(source_set)
    destination_set = Buffer(destination_set)

    set_source = cudf.Series(data=source_set, dtype=np.dtype("int32"))
    set_destination = cudf.Series(data=destination_set, dtype=np.dtype("int32"))
    #"""
    x= 1
    y= 1
    return x, y

