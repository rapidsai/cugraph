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

from cugraph.traversal.traveling_salesman cimport traveling_salesman as c_traveling_salesman
from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.graph_primtypes cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from numba import cuda

import cudf
import numpy as np


def traveling_salesman(pos_list,
                       restarts=4096,
                       beam_search=True,
                       k=4,
                       nstart=0,
                       verbose=False
):
    """
    Call traveling_salesman
    """

    nodes = pos_list.shape[0]
    cdef uintptr_t x_pos = <uintptr_t>NULL
    cdef uintptr_t y_pos = <uintptr_t>NULL

    pos_list['vertex'] = pos_list['vertex'].astype(np.int32)
    pos_list['x'] = pos_list['x'].astype(np.float32)
    pos_list['y'] = pos_list['y'].astype(np.float32)
    x_pos = pos_list['x'].__cuda_array_interface__['data'][0]
    y_pos = pos_list['y'].__cuda_array_interface__['data'][0]

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    cdef float final_cost_float = 0.0
    final_cost = None

    cdef uintptr_t route_ptr = <uintptr_t>NULL
    route_arr = cuda.device_array(nodes, dtype=np.int32)
    route_ptr = route_arr.device_ctypes_pointer.value

    cdef uintptr_t vtx_ptr = <uintptr_t>NULL
    vtx_ptr = pos_list['vertex'].__cuda_array_interface__['data'][0]

    final_cost_float = c_traveling_salesman(handle_[0],
            <int*> vtx_ptr,
            <int*> route_ptr,
            <float*> x_pos,
            <float*> y_pos,
            <int> nodes,
            <int> restarts,
            <bool> beam_search,
            <int> k,
            <int> nstart,
            <bool> verbose)

    route = cudf.Series(route_arr)
    final_cost = final_cost_float
    return route, final_cost
