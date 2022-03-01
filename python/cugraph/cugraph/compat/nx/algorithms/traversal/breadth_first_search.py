# Copyright (c) 2022, NVIDIA CORPORATION.
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
# limitations under the License.from networkx.algorithms.traversal import *
import cugraph


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    result = cugraph.bfs_edges(G, source, depth_limit, sort_neighbors)
    # convert to a list of tuples which the networkx algorithm returns
    formatted_result = result.query('distance > 0')[['distance', 'vertex']]
    formatted_result = formatted_result.to_records(index=False)
    np_array = map(tuple, formatted_result)
    list_of_tuples = tuple(np_array)
    return list_of_tuples
