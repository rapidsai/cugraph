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
from libc.stdint cimport uintptr_t

import cudf
import numpy as np
import cupy as cp


def traveling_salesman(pos_list,
                       restarts=4096,
                       k=4,
                       distance="euclidean"
):
    """
    Call traveling_salesman
    """

    nodes = pos_list.shape[0]
    cdef uintptr_t x_pos = <uintptr_t>NULL
    cdef uintptr_t y_pos = <uintptr_t>NULL

    if pos_list is not None:
        pos_list['x'] = pos_list['x'].astype(np.float32)
        pos_list['y'] = pos_list['y'].astype(np.float32)
        pos_list['x'][pos_list['vertex_id']] = pos_list['x']
        pos_list['y'][pos_list['vertex_id']] = pos_list['y']
        x_pos = pos_list['x'].__cuda_array_interface__['data'][0]
        y_pos = pos_list['y'].__cuda_array_interface__['data'][0]

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();
    cdef float final_cost_float = 0.0
    final_cost = None

    final_cost_float = c_traveling_salesman(handle_[0],
            <float*> x_pos,
            <float*> y_pos,
            <int> nodes,
            <int> restarts)
    final_cost = final_cost_float

    return final_cost
