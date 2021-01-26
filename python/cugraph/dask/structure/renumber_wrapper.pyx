#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
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
#

from cugraph.structure.utils_wrapper import *
import cudf
from cugraph.structure.graph_primtypes cimport *
import cugraph.structure.graph_primtypes_wrapper as graph_primtypes_wrapper
from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref
import numpy as np

from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer

def mg_renumber(input_df,
                num_global_verts,
                num_global_edges,    
                rank,
                handle):
    """
    Call MNMG renumber
    """
    cdef size_t handle_size_t = <size_t>handle.getHandle()

    src = input_df['src']
    dst = input_df['dst']
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL # set below...
    
    vertex_t = src.dtype
    if num_global_edges > (2**31 - 1):
        edge_t = np.dtype("int64")
    else:
        edge_t = np.dtype("int32")
    if "value" in input_df.columns:
        weights = input_df['value']
        weight_t = weights.dtype
        c_edge_weights = weights.__cuda_array_interface__['data'][0]
    else:
        weight_t = np.dtype("float32")

    # FIXME: needs to be edge_t type not int
    cdef int num_partition_edges = len(src)

    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]

    cdef bool is_hyper_partitioned = False # for now

    if (vertex_t == np.dtype("int32")):
        if ( edge_t == np.dtype("int32")):
            if( weight_t == np.dtype("float32")):
                maj_min_w = call_shuffle[int, int, float](handle,
                                                          c_src_vertices,
                                                          c_dst_vertices,
                                                          c_edge_weights,
                                                          num_partition_edges,
                                                          is_hyper_partitioned)
                # extract shuffled result:
                #
                cdef pair[unique_ptr[device_buffer], size_t] pair_s_major   = maj_min_w.get_major_wrap()
                cdef pair[unique_ptr[device_buffer], size_t] pair_s_minor   = maj_min_w.get_minor_wrap()
                cdef pair[unique_ptr[device_buffer], size_t] pair_s_weights = maj_min_w.get_weights_wrap()

                shufled_major_buffer = DeviceBuffer.c_from_unique_ptr(move(pair_s_major.first))
                shufled_major_buffer = Buffer(shufled_major_buffer)

                shufled_major_series = cudf.Series(data=shufled_major_buffer, dtype=vertex_t)

                shufled_minor_buffer = DeviceBuffer.c_from_unique_ptr(move(pair_s_minor.first))
                shufled_minor_buffer = Buffer(shufled_minor_buffer)

                shufled_minor_series = cudf.Series(data=shufled_minor_buffer, dtype=vertex_t)

                shufled_weights_buffer = DeviceBuffer.c_from_unique_ptr(move(pair_s_weights.first))
                shufled_weights_buffer = Buffer(shufled_weights_buffer)

                shufled_weights_series = cudf.Series(data=shufled_weights_buffer, dtype=weight_t)

                shuffled_df = cudf.DataFrame()
                shuffled_df['src']=shuffled_major_series
                shuffled_df['dst']=shuffled_minor_series
                shuffled_df['weights']= shuffled_weights_series
                        
                cdef vertex_t* shuffled_major = shufled_major_series.__cuda_array_interface__['data'][0]
                cdef vertex_t* shuffled_minor = shufled_minor_series.__cuda_array_interface__['data'][0]
                
                cdef bool do_check = False # ? for now...
                cdef bool mg_flag = True # run MNMG
                
                renum_quad = call_renumber[int, int](handle,
                                                     shuffled_major,
                                                     shuffled_minor,
                                                     num_partition_edges,
                                                     is_hyper_partitioned,
                                                     do_check,
                                                     mg_flag)
