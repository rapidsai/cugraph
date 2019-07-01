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

import cugraph.community.louvain_wrapper as cpp_louvain


def louvain(input_graph):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain heuristic

    Parameters
    ----------
    input_graph : cuGraph.Graph
      cuGraph graph descriptor, should contain the connectivity information as
      an edge list.
      The adjacency list will be computed if not already present. The graph
      should be undirected where an undirected edge is represented by a
      directed edge in both direction.

    Returns
    -------
    parts, modularity_score  : cudf.DataFrame
      parts: GPU data frame of size V containing two columns: the
          vertex id and the partition id it is assigned to.
      modularity_score: a double value containing the modularity score of the
          partitioning

    Examples
    --------
    >>> M = read_mtx_file(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> parts, modularity_score = cuGraph.louvain(G)
    """

    parts, modularity_score = cpp_louvain.louvain(input_graph.graph_ptr)

    return parts, modularity_score
