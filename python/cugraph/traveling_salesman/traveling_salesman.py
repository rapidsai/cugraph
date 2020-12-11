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
                       node_pos=None,
                       restarts=4096,
                       weight=None
                       distance="euclidean",

):

    if node_pos is not None:
        null_check(node_pos['vertex'])
        null_check(node_pos['x'])
        null_check(node_pos['y'])
        if input_graph.renumbered is True:
            node_pos = input_graph.add_internal_vertex_id(node_pos,
                                                "vertex",
                                                "vertex")

    if input_graph.is_directed():
        input_graph.to_undirected()

    cost = tsp_wrapper.traveling_salesman(input_graph,
                                          node_pos,
                                          restarts,
                                          weight,
                                          distance)

    if input_graph.renumbered:
        node_pos = input_graph.unrenumber(node_pos, "vertex")

    return cost
