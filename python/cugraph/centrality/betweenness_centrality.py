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

import random
import numpy as np
from cugraph.centrality import betweenness_centrality_wrapper


# NOTE: result_type=float could ne an intuitive way to indicate the result type
def betweenness_centrality(G, k=None, normalized=True,
                           weight=None, endpoints=False, implementation=None,
                           seed=None, result_dtype=np.float32):
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

    k : int or list or None, optional, default=None
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation
        If k is a list, use the content of the list for estimation

    normalized : bool, optional, default=True
        Value defaults to true.  If true, the betweenness values are normalized
        by 2/((n-1)(n-2)) for graphs, and 1 / ((n-1)(n-2)) for directed graphs
        where n is the number of nodes in G.

    weight : cudf.Series, optional, default=None
        Specifies the weights to be used for each vertex.

    endpoints : bool, optional, default=False
        If true, include the endpoints in the shortest path counts

    implementation : string, optional, default=None
        if implementation is None or "default", uses native cugraph,
        if "gunrock" uses gunrock based bc

    seed : optional
        if k is specified and seed is not None, use seed to initialize the
        random number generator

    result_dtype : np.float32 or np.float64, optional, default=np.float32
        Indicate the data type of the betweenness centrality scores
        Using double automatically switch implementation to default

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding betweenness centrality values.
        Please note that the resulting the 'vertex' column might not be
        in ascending order.

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
    if implementation is None:
        implementation = "default"
    if implementation not in ["default", "gunrock"]:
        raise Exception("Only two implementations are supported: 'default' "
                        "and 'gunrock'")

    if k is not None:
        if implementation == "gunrock":
            raise Exception("sampling feature of betweenness "
                            "centrality not currently supported "
                            "with gunrock implementation, "
                            "please use None or 'default'")
        # In order to compare with preset sources,
        # k can either be a list or an integer or None
        #  int: Generate an random sample with k elements
        # list: k become the length of the list and vertices become the content
        # None: All the vertices are considered
        if isinstance(k, int):
            random.seed(seed)
            vertices = random.sample(range(G.number_of_vertices()), k)
        # Using k as a list allows to have an easier way to compare against
        # other implementations on
        elif isinstance(k, list):
            vertices = k
            k = len(vertices)
        # FIXME: There might be a cleaner way to obtain the inverse mapping
        if G.renumbered:
            vertices = [G.edgelist.renumber_map[G.edgelist.renumber_map ==
                                                vert].index[0] for vert in
                        vertices]

    if endpoints is not False:
        raise NotImplementedError("endpoints accumulation for betweenness "
                                  "centrality not currently supported")

    if weight is not None:
        raise NotImplementedError("weighted implementation of betweenness "
                                  "centrality not currently supported")
    if result_dtype not in [np.float32, np.float64]:
        raise TypeError("result type can only be float or double centrality "
                        "not currently supported")

    df = betweenness_centrality_wrapper.betweenness_centrality(G, normalized,
                                                               endpoints,
                                                               weight,
                                                               k, vertices,
                                                               implementation,
                                                               result_dtype)
    return df
