# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import cudf
from cugraph.centrality import betweenness_centrality_wrapper
from cugraph.centrality import edge_betweenness_centrality_wrapper
from cugraph.utilities import (
    df_edge_score_to_dictionary,
    df_score_to_dictionary,
    ensure_cugraph_obj_for_nx,
)


# NOTE: result_type=float could be an intuitive way to indicate the result type
def betweenness_centrality(
    G,
    k=None,
    normalized=True,
    weight=None,
    endpoints=False,
    seed=None,
    result_dtype=np.float64,
):
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    To improve performance. rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    CuGraph does not currently support the 'endpoints' and 'weight' parameters
    as seen in the corresponding networkX call.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        Weights in the graph are ignored, the current implementation uses
        BFS traversals. Use weight parameter if weights need to be considered
        (currently not supported)

    k : int or list or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.  If k is a list, use the content
        of the list for estimation: the list should contain vertex
        identifiers. If k is None (the default), all the vertices are used
        to estimate betweenness.  Vertices obtained through sampling or
        defined as a list will be used assources for traversals inside the
        algorithm.

    normalized : bool, optional (default=True)
        If true, the betweenness values are normalized by
        __2 / ((n - 1) * (n - 2))__ for undirected Graphs, and
        __1 / ((n - 1) * (n - 2))__ for directed Graphs
        where n is the number of nodes in G.
        Normalization will ensure that values are in [0, 1],
        this normalization scales for the highest possible value where one
        node is crossed by every single shortest path.

    weight : cudf.DataFrame, optional (default=None)
        Specifies the weights to be used for each edge.
        Should contain a mapping between
        edges and weights.
        (Not Supported)

    endpoints : bool, optional (default=False)
        If true, include the endpoints in the shortest path counts.
        (Not Supported)

    seed : optional (default=None)
        if k is specified and k is an integer, use seed to initialize the
        random number generator.
        Using None as seed relies on random.seed() behavior: using current
        system time
        If k is either None or list: seed parameter is ignored

    result_dtype : np.float32 or np.float64, optional, default=np.float64
        Indicate the data type of the betweenness centrality scores

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding betweenness centrality values.
        Please note that the resulting the 'vertex' column might not be
        in ascending order.  The Dictionary contains the same two columns

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['betweenness_centrality'] : cudf.Series
            Contains the betweenness centrality of vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> bc = cugraph.betweenness_centrality(G)

    """
    # vertices is intended to be a cuDF series that contains a sampling of
    # k vertices out of the graph.
    #
    # NOTE: cuDF doesn't currently support sampling, but there is a python
    # workaround.

    if weight is not None:
        raise NotImplementedError(
            "weighted implementation of betweenness "
            "centrality not currently supported"
        )

    if result_dtype not in [np.float32, np.float64]:
        raise TypeError("result type can only be np.float32 or np.float64")

    G, isNx = ensure_cugraph_obj_for_nx(G)

    vertices = _initialize_vertices(G, k, seed)

    df = betweenness_centrality_wrapper.betweenness_centrality(
        G, normalized, endpoints, weight, vertices, result_dtype
    )

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "betweenness_centrality")
        return dict
    else:
        return df


def edge_betweenness_centrality(
    G, k=None, normalized=True, weight=None, seed=None, result_dtype=np.float64
):
    """
    Compute the edge betweenness centrality for all edges of the graph G.
    Betweenness centrality is a measure of the number of shortest paths
    that pass over an edge.  An edge with a high betweenness centrality
    score has more paths passing over it and is therefore believed to be
    more important.

    To improve performance, rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    CuGraph does not currently support the 'weight' parameter
    as seen in the corresponding networkX call.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        Weights in the graph are ignored, the current implementation uses
        BFS traversals. Use weight parameter if weights need to be considered
        (currently not supported)

    k : int or list or None, optional (default=None)
        If k is not None, use k node samples to estimate betweenness.  Higher
        values give better approximation.
        If k is a list, use the content of the list for estimation: the list
        should contain vertices identifiers.
        Vertices obtained through sampling or defined as a list will be used as
        sources for traversals inside the algorithm.

    normalized : bool, optional (default=True)
        If true, the betweenness values are normalized by
        2 / (n * (n - 1)) for undirected Graphs, and
        1 / (n * (n - 1)) for directed Graphs
        where n is the number of nodes in G.
        Normalization will ensure that values are in [0, 1],
        this normalization scales for the highest possible value where one
        edge is crossed by every single shortest path.

    weight : cudf.DataFrame, optional (default=None)
        Specifies the weights to be used for each edge.
        Should contain a mapping between
        edges and weights.
        (Not Supported)

    seed : optional (default=None)
        if k is specified and k is an integer, use seed to initialize the
        random number generator.
        Using None as seed relies on random.seed() behavior: using current
        system time
        If k is either None or list: seed parameter is ignored

    result_dtype : np.float32 or np.float64, optional (default=np.float64)
        Indicate the data type of the betweenness centrality scores
        Using double automatically switch implementation to "default"

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing three cudf.Series of size E: the vertex
        identifiers of the sources, the vertex identifies of the destinations
        and the corresponding betweenness centrality values.
        Please note that the resulting the 'src', 'dst' column might not be
        in ascending order.

        df['src'] : cudf.Series
            Contains the vertex identifiers of the source of each edge

        df['dst'] : cudf.Series
            Contains the vertex identifiers of the destination of each edge

        df['edge_betweenness_centrality'] : cudf.Series
            Contains the betweenness centrality of edges

        When using undirected graphs, 'src' and 'dst' only contains elements
        such that 'src' < 'dst', which might differ from networkx and user's
        input. Namely edge (1 -> 0) is transformed into (0 -> 1) but
        contains the betweenness centrality of edge (1 -> 0).


    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> bc = cugraph.edge_betweenness_centrality(G)

    """
    if weight is not None:
        raise NotImplementedError(
            "weighted implementation of edge betweenness "
            "centrality not currently supported"
        )
    if result_dtype not in [np.float32, np.float64]:
        raise TypeError("result type can only be np.float32 or np.float64")

    G, isNx = ensure_cugraph_obj_for_nx(G)
    vertices = _initialize_vertices(G, k, seed)

    df = edge_betweenness_centrality_wrapper.edge_betweenness_centrality(
        G, normalized, weight, vertices, result_dtype
    )

    if G.renumbered:
        df = G.unrenumber(df, "src")
        df = G.unrenumber(df, "dst")

    if G.is_directed() is False:
        # select the lower triangle of the df based on src/dst vertex value
        lower_triangle = df["src"] >= df["dst"]
        # swap the src and dst vertices for the lower triangle only. Because
        # this is a symmeterized graph, this operation results in a df with
        # multiple src/dst entries.
        df["src"][lower_triangle], df["dst"][lower_triangle] = (
            df["dst"][lower_triangle],
            df["src"][lower_triangle],
        )
        # overwrite the df with the sum of the values for all alike src/dst
        # vertex pairs, resulting in half the edges of the original df from the
        # symmeterized graph.
        df = df.groupby(by=["src", "dst"]).sum().reset_index()

    if isNx is True:
        return df_edge_score_to_dictionary(df, "betweenness_centrality")
    else:
        return df


# In order to compare with pre-set sources,
# k can either be a list or an integer or None
#  int: Generate an random sample with k elements
# list: k become the length of the list and vertices become the content
# None: All the vertices are considered
def _initialize_vertices(G, k, seed):
    vertices = None
    numpy_vertices = None
    if k is not None:
        if isinstance(k, int):
            vertices = _initialize_vertices_from_indices_sampling(G, k, seed)
        elif isinstance(k, list):
            vertices = _initialize_vertices_from_identifiers_list(G, k)
        numpy_vertices = np.array(vertices, dtype=np.int32)
    else:
        numpy_vertices = np.arange(G.number_of_vertices(), dtype=np.int32)
    return numpy_vertices


# NOTE: We do not renumber in case k is an int, the sampling is
#       not operating on the valid vertices identifiers but their
#       indices:
# Example:
# - vertex '2' is missing
# - vertices '0' '1' '3' '4' exist
# - There is a vertex at index 2 (there is not guarantee that it is
#   vertice '3' )
def _initialize_vertices_from_indices_sampling(G, k, seed):
    random.seed(seed)
    vertices = random.sample(range(G.number_of_vertices()), k)
    return vertices


def _initialize_vertices_from_identifiers_list(G, identifiers):
    vertices = identifiers
    if G.renumbered:
        vertices = G.lookup_internal_vertex_id(cudf.Series(vertices)).to_numpy()

    return vertices
