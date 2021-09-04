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

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr

from pylibcugraph.components._connectivity cimport *


def weakly_connected_components(src, dst, weights, num_verts, num_edges, labels):

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    cdef uintptr_t c_src_vertices    = src['data'][0]
    cdef uintptr_t c_dst_vertices    = dst['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    cdef uintptr_t c_labels_val = labels['data'][0]

    cdef graph_container_t graph_container
    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>NULL,
                             <void*>NULL,
                             0,
                             <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                             <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                             <numberTypeEnum>(<int>(numberTypeEnum.floatType)),
                             num_edges,
                             num_verts, num_edges,
                             False,
                             True,
                             False,
                             False)

    call_wcc[int, float](handle_ptr.get()[0],
                         graph_container,
                         <int*> c_labels_val)


def strongly_connected_components(offsets, indices, weights, num_verts, num_edges, labels):
    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    cdef uintptr_t c_offsets    = offsets['data'][0]
    cdef uintptr_t c_indices    = indices['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    cdef uintptr_t c_labels_val = labels['data'][0]

    cdef GraphCSRView[int,int,float] g

    g = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices, <float*>NULL, num_verts, num_edges)

    cdef cugraph_cc_t connect_type=CUGRAPH_STRONG
    connected_components(g, <cugraph_cc_t>connect_type, <int *>c_labels_val)
