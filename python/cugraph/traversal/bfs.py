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

from cugraph.traversal import bfs_wrapper
from cugraph.structure.graph import Graph
from cugraph.utilities import check_nx_graph


def bfs(G, start, return_sp_counter=False):
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

    return_sp_counter : bool, optional, default=False
        Indicates if shortest path counters should be returned

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex

        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex

        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

        df['sp_counter'][i] gives for the i'th vertex the number of shortest
        path leading to it during traversal (Only if retrun_sp_counter is True)

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.bfs(G, 0)
    """

    if type(G) is Graph:
        directed = False
    else:
        directed = True

    if G.renumbered is True:
        start = G.lookup_internal_vertex_id(cudf.Series([start]))[0]

    df = bfs_wrapper.bfs(G, start, directed, return_sp_counter)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")
        df = G.unrenumber(df, "predecessor")
        df["predecessor"].fillna(-1, inplace=True)

    return df


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None,
              return_sp_counter=False):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.

    Parameters
    ----------
    G : cugraph.graph or NetworkX.Graph
        graph descriptor that contains connectivity information
    source : Integer
        The starting vertex index
    reverse : boolean
        If a directed graph, then process edges in a reverse direction
        Currently not implemented
    depth_limit : Int or None
        Limit the depth of the search
        Currently not implemented
    sort_neighbors : None or Function
        Currently not implemented
    return_sp_counter : bool, optional, default=False
        Indicates if shortest path counters should be returned

    Returns
    -------
    df : cudf.DataFrame or Pandas.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex

        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex

        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

        df['sp_counter'][i] gives for the i'th vertex the number of shortest
        path leading to it during traversal (Only if retrun_sp_counter is True)

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.bfs_edges(G, 0)
    """

    if reverse is not None:
        raise NotImplementedError(
            "reverse processing of graph is "
            "currently not supported"
        )

    if depth_limit is not None:
        raise NotImplementedError(
            "depth limit implementation of BFS "
            "is not currently supported"
        )

    G, isNx = check_nx_graph(G)

    df = bfs(G, source, return_sp_counter)

    if isNx is True:
        df = df.to_pandas()

    return df
