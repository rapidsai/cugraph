# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

import cudf
from cugraph.traversal.bfs import bfs
from cugraph.traversal.bfs import bfs_edges
from cugraph.traversal.sssp import (
    sssp,
    shortest_path,
    filter_unreachable
)


# TODO: need to add docs, test with cupy matrix and networkx graphs
def shortest_path_length(G, source, target=None):
    df = sssp(G, source)
    if target is not None:
        if not G.has_node(target):
            raise ValueError("Graph does not contain target vertex")
        target_distance = df.loc[df["vertex"] == target]
        return target_distance.iloc[0]["distance"]
    else:
        results = cudf.DataFrame()
        results["vertex"] = df["vertex"]
        results["distance"] = df["distance"]
        return results


__all__ = [
    "bfs",
    "bfs_edges",
    "sssp",
    "shortest_path",
    "shortest_path_length",
    "filter_unreachable"
]
