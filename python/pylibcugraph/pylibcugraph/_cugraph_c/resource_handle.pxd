# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

cdef extern from "cugraph_c/resource_handle.h":

    ctypedef struct cugraph_resource_handle_t:
        pass

    # FIXME: the void* raft_handle arg will change in a future release
    cdef cugraph_resource_handle_t* \
        cugraph_create_resource_handle(
	    void* raft_handle
	)

    cdef int \
        cugraph_resource_handle_get_rank(
	    const cugraph_resource_handle_t* handle
	)

    cdef void \
        cugraph_free_resource_handle(
            cugraph_resource_handle_t* p_handle
        )
