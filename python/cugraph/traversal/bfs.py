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

from cugraph.traversal import bfs_wrapper


def bfs(G, start, directed=True):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.

    Parameters
    ----------
    G : cugraph.graph
        cuGraph graph descriptor, should contain the connectivity information
        as an adjacency list.
    start : Integer
        The index of the graph vertex from which the traversal begins
    directed : bool
        Indicates whether the graph in question is a directed graph, or whether
        each edge has a corresponding reverse edge. (Allows optimizations if
        the graph is undirected)

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex

        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex

        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> df = cugraph.bfs(G, 0)
    """

    df = bfs_wrapper.bfs(G, start, directed)

    return df
