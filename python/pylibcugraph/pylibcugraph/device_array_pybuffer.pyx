# Copyright (c) 2022, NVIDIA CORPORATION.
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

# Have cython use python 3 syntax
# cython: language_level = 3

from cpython cimport Py_buffer


cdef class DeviceArrayPyBuffer:
    cdef Py_ssize_t ncols
    cdef Py_ssize_t shape
    cdef Py_ssize_t strides

    def __cinit__(
        self,
        cugraph_resource_handle_t* c_resource_handle_ptr,
        cugraph_type_erased_device_array_view_t* device_array_view_ptr):
	self.__resource_handle = c_resource_handle_ptr
	self.__device_array_view = device_array_view_ptr


    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(self.v[0])

        self.shape[0] = self.v.size() / self.ncols
        self.shape[1] = self.ncols

        # Stride 1 is the distance, in bytes, between two items in a row;
        # this is the distance between two adjacent items in the vector.
        # Stride 0 is the distance between the first elements of adjacent rows.
        self.strides[1] = <Py_ssize_t>(  <char *>&(self.v[1])
                                       - <char *>&(self.v[0]))
        self.strides[0] = self.ncols * self.strides[1]

        buffer.buf = <char *>&(self.v[0])
        buffer.format = 'f'                     # float
        buffer.internal = NULL                  # see References
        buffer.itemsize = itemsize
        buffer.len = self.v.size() * itemsize   # product(shape) * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 0
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL                # for pointer arrays only

    def __releasebuffer__(self, Py_buffer *buffer):
        pass
