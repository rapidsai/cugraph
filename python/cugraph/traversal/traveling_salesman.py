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
                       beam_search=True,
                       k=4,
                       nstart=0,
                       verbose=False,
):
    if not isinstance(pos_list, cudf.DataFrame):
        raise Exception("Instance should be cudf.DataFrame")

    null_check(pos_list['vertex'])
    null_check(pos_list['x'])
    null_check(pos_list['y'])

    route, cost = traveling_salesman_wrapper.traveling_salesman(pos_list,
                                                                restarts,
                                                                beam_search,
                                                                k,
                                                                nstart,
                                                                verbose)
    return route, cost
