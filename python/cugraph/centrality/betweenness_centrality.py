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

from cugraph.centrality import betweenness_centrality_wrapper


def betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    """
    Compute betweenness centrality for the nodes of the graph G. cuGraph
    does not currently support the 'endpoints' and 'weight' parameters
    as seen in the corresponding networkX call.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges where undirected edges are
        represented as directed edges in both directions.
    k : int, optional
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation
    normalized : bool, optional
        Value defaults to true.  If true, the betweenness values are normalized
        by 2/((n-1)(n-2)) for graphs, and 1 / ((n-1)(n-2)) for directed graphs
        where n is the number of nodes in G.
    weight : cudf.Series
        Specifies the weights to be used for each vertex.
    endpoints : bool, optional
        If true, include the endpoints in the shortest path counts
    seed : optional
        k is specified and seed is not None, use seed to initialize the random
        number generator

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding katz centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['betweenness_centrality'] : cudf.Series
            Contains the betweenness centrality of vertices

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> bc = cugraph.betweenness_centrality(G)
    """

    #
    # Some features not implemented for gunrock implementation, failing fast,
    # but passing parameters through
    #
    # vertices is intended to be a cuDF series that contains a sampling of
    # k vertices out of the graph.
    #
    # NOTE: cuDF doesn't currently support sampling, but there is a python
    # workaround.
    #
    vertices = None
    if k is not None:
        raise Exception("sampling feature of betweenness centrality not currently supported")

    if weight is not None:
        raise Exception("weighted implementation of betweenness centrality not currently supported")

    df = betweenness_centrality_wrapper.betweenness_centrality(G, normalized, endpoints, weight, k, vertices)
    return df
