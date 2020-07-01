# Copyright (c) 2019 - 2020, NVIDIA CORPORATION.
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

from cugraph.community import louvain_wrapper
from cugraph.structure.graph import Graph


def louvain(input_graph, max_iter=100):
    """
    Compute the modularity optimizing partition of the input graph using the
    Louvain heuristic

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph descriptor of type Graph

        The adjacency list will be computed if not already present. The graph
        should be undirected where an undirected edge is represented by a
        directed edge in both direction.

    max_iter : integer
        This controls the maximum number of levels/iterations of the Louvain
        algorithm. When specified the algorithm will terminate after no more
        than the specified number of iterations. No error occurs when the
        algorithm terminates early in this manner.

    Returns
    -------
    parts : cudf.DataFrame
        GPU data frame of size V containing two columns the vertex id and the
        partition id it is assigned to.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['partition'] : cudf.Series
            Contains the partition assigned to the vertices

    modularity_score : float
        a floating point number containing the global modularity score of the
        partitioning.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv',
                          delimiter = ' ',
                          dtype=['int32', 'int32', 'float32'],
                          header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> parts, modularity_score = cugraph.louvain(G)
    """

    if type(input_graph) is not Graph:
        raise Exception("input graph must be undirected")

    parts, modularity_score = louvain_wrapper.louvain(
        input_graph, max_iter=max_iter
    )

    if input_graph.renumbered:
        # FIXME: multi-column vertex support
        parts = input_graph.edgelist.renumber_map.from_vertex_id(
            parts, "vertex", drop=True
        ).rename({"0": "vertex"})

    return parts, modularity_score
