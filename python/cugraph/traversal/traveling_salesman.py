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

from cugraph.algorithms import tsp_wrapper
from cugraph.structure.graph import null_check

def traveling_salesman(input_graph,
                       pos_list=None,
                       restarts=4096,
                       distance="euclidean",

):
    if type(G) is not Graph:
        raise Exception("input graph must be undirected")

    if distance != "euclidean":
        raise Exception("Other metrics not supported")

    if pos_list is not None:
        null_check(pos_list['vertex'])
        null_check(pos_list['x'])
        null_check(pos_list['y'])
        if input_graph.renumbered is True:
            pos_list = input_graph.add_internal_vertex_id(pos_list,
                                                "vertex",
                                                "vertex")

    cost = tsp_wrapper.traveling_salesman(input_graph,
                                          pos_list,
                                          restarts,
                                          weight,
                                          distance)

    if input_graph.renumbered:
        pos_list = input_graph.unrenumber(pos_list, "vertex")

    return cost
