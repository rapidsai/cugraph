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

from cugraph.traversal import traveling_salesman_wrapper
from cugraph.structure.graph import null_check
from cugraph.structure.number_map import NumberMap
import cudf

def traveling_salesman(pos_list,
                       restarts=4096,
                       k=4,
                       distance="euclidean",
                       verbose=False
):
    if not isinstance(pos_list, cudf.DataFrame):
        raise Exception("Instance should be cudf.DataFrame")

    if distance != "euclidean":
        raise Exception("Metric not supported")

    null_check(pos_list['vertex'])
    null_check(pos_list['x'])
    null_check(pos_list['y'])

    # Renumber
    numbering = NumberMap()
    numbering.from_series(pos_list['vertex'])
    pos_list = numbering.add_internal_vertex_id(pos_list,
                                                'vertex_id',
                                                'vertex',
                                                drop=False,
                                                preserve_order=True)

    cost = traveling_salesman_wrapper.traveling_salesman(pos_list,
                                                         restarts,
                                                         k,
                                                         distance,
                                                         verbose)
    # Drop internal ids and generated column
    pos_list = pos_list[["vertex", "x", "y"]]

    return cost
