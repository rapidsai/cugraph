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
from cugraph.utilities.graph_generator cimport *
#from cugraph.utilities.graph_generator cimport call_generate_rmat_edgelists
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
from cugraph.structure.utils_wrapper import *
from cugraph.dask.utilities cimport mg_graph_generator_edgelist as c_graph_generator
import cugraph.structure.graph_primtypes_wrapper as graph_primtypes_wrapper




def mg_graph_generator_edgelist(
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
    if num_edges > (2**31 - 1):
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

