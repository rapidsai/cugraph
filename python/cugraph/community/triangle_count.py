# Copyright (c) 2019, NVIDIA CORPORATION.
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

import cugraph
import cugraph.community.triangle_count_wrapper as cpp_triangle_count


def triangles(G):
    """
    Compute the triangle (number of cycles of length three) count of the 
    input graph.
    
    Parameters
    ----------
    G : cugraph.graph
      cugraph graph descriptor, should contain the connectivity information,
      (edge weights are not used in this algorithm)
    
    Returns
    -------
    count : A 64 bit integer whose value gives the number of triangles in the
      graph.
      
    Example
    -------
    >>>> M = read_mtx_file(graph_file)
    >>>> sources = cudf.Series(M.row)
    >>>> destinations = cudf.Series(M.col)
    >>>> G = cugraph.Graph()
    >>>> G.add_edge_list(sources, destinations, None)
    >>>> count = cugraph.triangle_count(G)
    """

    result = cpp_triangle_count.triangles(G.graph_ptr)

    return result
