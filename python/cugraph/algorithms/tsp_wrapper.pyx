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

def tsp(input_graph):
    """
    Call tsp
    """

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df['src'])

    cdef GraphCOOView[int,int,float] graph_float
    cdef GraphCOOView[int,int,double] graph_double

    cdef uintptr_t c_src_indices = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if input_graph.edgelist.weights:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]



    return df
