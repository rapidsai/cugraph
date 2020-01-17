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

from cugraph.components import connectivity_wrapper


def weakly_connected_components(G):
    """
    Generate the weakly connected components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        Currently, the graph should be undirected where an undirected edge is
        represented by a directed edge in both directions. The adjacency list
        will be computed if not already present. The number of vertices should
        fit into a 32b int.

    Returns
    -------
    df : cudf.DataFrame
      df['labels'][i] gives the label id of the i'th vertex
      df['vertices'][i] gives the vertex id of the i'th vertex

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> df = cugraph.weakly_connected_components(G)
    """

    df = connectivity_wrapper.weakly_connected_components(G)

    return df


def strongly_connected_components(G):
    """
    Generate the stronlgly connected components and attach a component label to
    each vertex.

    Parameters
    ----------
    G : cugraph.Graph
      cuGraph graph descriptor, should contain the connectivity information as
      an edge list (edge weights are not used for this algorithm). The graph
      can be either directed or undirected where an undirected edge is
      represented by a directed edge in both directions.
      The adjacency list will be computed if not already present.
      The number of vertices should fit into a 32b int.

    Returns
    -------
    df : cudf.DataFrame
      df['labels'][i] gives the label id of the i'th vertex
      df['vertices'][i] gives the vertex id of the i'th vertex

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> df = cugraph.strongly_connected_components(G)
    """

    df = connectivity_wrapper.strongly_connected_components(G)

    return df
