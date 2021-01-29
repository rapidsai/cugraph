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

from libcpp.vector cimport vector
from libcpp.utility cimport move
from rmm._lib.device_buffer cimport device_buffer, DeviceBuffer


# class needed for creating a cudf.Series()
# from a std::vector; this is adapted from cudf
# by making a template class out of it;
# Note: Cython doesn't compile template classes
# that are not `extern` from C++;
# It also does not compile a fused type;
# Hence, only option left is to create
# separate classes for each underlying
# `element_type` 
#
#
# Wrapper around vector<int32_t>:
#
cdef class BufferArrayFromInt32Vector:
    cdef Py_ssize_t length
    cdef unique_ptr[vector[int]] in_vec

    # these two things declare part of the buffer interface
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    @staticmethod
    cdef BufferArrayFromInt32Vector from_unique_ptr(
        unique_ptr[vector[int]] in_vec
    ):
        cdef BufferArrayFromInt32Vector buf = BufferArrayFromInt32Vector()
        buf.in_vec = move(in_vec)
        buf.length = deref(buf.in_vec).size()
        return buf

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(int)

        self.shape[0] = self.length
        self.strides[0] = 1

        buffer.buf = deref(self.in_vec).data()

        buffer.format = NULL  # byte
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.length * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


# Wrapper around vector<int64_t>:
#
cdef class BufferArrayFromInt64Vector:
    cdef Py_ssize_t length
    cdef unique_ptr[vector[long]] in_vec

    # these two things declare part of the buffer interface
    cdef Py_ssize_t shape[1]
    cdef Py_ssize_t strides[1]

    @staticmethod
    cdef BufferArrayFromInt64Vector from_unique_ptr(
        unique_ptr[vector[long]] in_vec
    ):
        cdef BufferArrayFromInt64Vector buf = BufferArrayFromInt64Vector()
        buf.in_vec = move(in_vec)
        buf.length = deref(buf.in_vec).size()
        return buf

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(long)

        self.shape[0] = self.length
        self.strides[0] = 1

        buffer.buf = deref(self.in_vec).data()

        buffer.format = NULL  # byte
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.length * itemsize   # product(shape) * itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


cdef renumber_helper(shuffled_vertices_t* ptr_maj_min_w):
    # extract shuffled result:
    #
    cdef pair[unique_ptr[device_buffer], size_t] pair_s_major   = deref(ptr_maj_min_w).get_major_wrap()
    cdef pair[unique_ptr[device_buffer], size_t] pair_s_minor   = deref(ptr_maj_min_w).get_minor_wrap()
    cdef pair[unique_ptr[device_buffer], size_t] pair_s_weights = deref(ptr_maj_min_w).get_weights_wrap()
    
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
    
    return shuffled_df

def mg_renumber(input_df,           # maybe use cpdef ?
                num_global_verts,
                num_global_edges,    
                rank,
                handle):
    """
    Call MNMG renumber
    """
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    # TODO: get handle_t out of handle...
    handle_ptr = <handle_t*>handle_size_t

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

    cdef uintptr_t shuffled_major = <uintptr_t>NULL
    cdef uintptr_t shuffled_minor = <uintptr_t>NULL
    
    cdef bool do_check = False # ? for now...
    cdef bool mg_flag = True # run MNMG

    cdef pair[unique_ptr[device_buffer], size_t] pair_original
    cdef pair[unique_ptr[device_buffer], size_t] pair_partition

    cdef unique_ptr[major_minor_weights_t[int, float]] ptr_shuffled_32_32
    cdef unique_ptr[major_minor_weights_t[int, double]] ptr_shuffled_32_64

    cdef unique_ptr[renum_quad_t[int, int]] ptr_renum_quad_32_32
    cdef unique_ptr[renum_quad_t[int, long]] ptr_renum_quad_32_64
    cdef unique_ptr[renum_quad_t[long, long]] ptr_renum_quad_64_64
    
    if (vertex_t == np.dtype("int32")):
        if ( edge_t == np.dtype("int32")):
            if( weight_t == np.dtype("float32")):
                ptr_shuffled_32_32.reset(call_shuffle[int, int, float](deref(handle_ptr),
                                                                       <int*>c_src_vertices,
                                                                       <int*>c_dst_vertices,
                                                                       <float*>c_edge_weights,
                                                                       num_partition_edges,
                                                                       is_hyper_partitioned).release())
                
                shuffled_df = renumber_helper(ptr_shuffled_32_32.get())
                
                shuffled_src = shufled_df['src']
                shuffled_dst = shufled_df['dst']
                        
                shuffled_major = shuffled_src.__cuda_array_interface__['data'][0]
                shuffled_minor = shuffled_dst.__cuda_array_interface__['data'][0]
                
                ptr_renum_quad_32_32.reset(call_renumber[int, int](deref(handle_ptr),
                                                                   <int*>shuffled_major,
                                                                   <int*>shuffled_minor,
                                                                   num_partition_edges,
                                                                   is_hyper_partitioned,
                                                                   do_check,
                                                                   mg_flag).release())
                
                pair_original = ptr_renum_quad_32_32.get().get_dv_wrap() # original vertices: see helper
                

                original_buffer = DeviceBuffer.c_from_unique_ptr(move(pair_original.first))
                original_buffer = Buffer(original_buffer)

                original_series = cudf.Series(data=original_buffer, dtype=vertex_t)

                # FIXME: this is a std::vector<>; cannot be converted to a DeviceBuffer:
                #
                # pair_partition = ptr_renum_quad_32_32.get().get_partition_offsets()
                
                # partition_buffer = DeviceBuffer.c_from_unique_ptr(move(pair_partition.first))
                # partition_buffer = Buffer(partition_buffer)

                # # create series
                # #
                # new_series = cudf.Series(np.arange(partition_buffer.iloc[rank],
                #                                    partition_buffer.iloc[rank+1]),
                #                          dtype=vertex_t)
                
                # # create new cudf df
                # #
                # # and add the previous series to it:
                # #
                # renumbered_map = cudf.DataFrame()
                # renumbered_map['original_ids'] = original_series
                # renumbered_map['new_ids'] = new_series

                # return renumbered_map, shuffled_df
