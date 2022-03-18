#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.memory cimport shared_ptr
from rmm._lib.cuda_stream_view cimport cuda_stream_view
from rmm._lib.cuda_stream_pool cimport cuda_stream_pool
from libcpp.memory cimport shared_ptr
from libcpp.memory cimport unique_ptr

cdef extern from "raft/handle.hpp" namespace "raft" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(cuda_stream_view stream_view) except +
        handle_t(cuda_stream_view stream_view,
                 shared_ptr[cuda_stream_pool] stream_pool) except +
        cuda_stream_view get_stream() except +
        void sync_stream() except +

cdef class Handle:
    cdef unique_ptr[handle_t] c_obj
    cdef shared_ptr[cuda_stream_pool] stream_pool
    cdef int n_streams
