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

import cudf
from cugraph.structure.graph_classes import Graph
from cugraph.link_prediction import jaccard_wrapper
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_edge_score_to_dictionary,
                               renumber_vertex_pair,
                               )


def jaccard(input_graph, vertex_pair=None):
    """
    Compute the Jaccard similarity between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Jaccard similarity is defined between two sets as the ratio of the volume
    of their intersection divided by the volume of their union. In the context
    of graphs, the neighborhood of a vertex is seen as a set. The Jaccard
    similarity weight of each edge represents the strength of connection
    between vertices based on the relative similarity of their neighbors. If
    first is specified but second is not, or vice versa, an exception will be
    thrown.

    NOTE: If the vertex_pair parameter is not specified then the behavior
    of cugraph.jaccard is different from the behavior of
    networkx.jaccard_coefficient.

    cugraph.jaccard, in the absence of a specified vertex pair list, will
    use the edges of the graph to construct a vertex pair list and will
    return the jaccard coefficient for those vertex pairs.

    networkx.jaccard_coefficient, in the absence of a specified vertex
    pair list, will return an upper triangular dense matrix, excluding
    the diagonal as well as vertex pairs that are directly connected
    by an edge in the graph, of jaccard coefficients.  Technically, networkx
    returns a lazy iterator across this upper triangular matrix where
    the actual jaccard coefficient is computed when the iterator is
    dereferenced.  Computing a dense matrix of results is not feasible
    if the number of vertices in the graph is large (100,000 vertices
    would result in 4.9 billion values in that iterator).

    If your graph is small enough (or you have enough memory and patience)
    you can get the interesting (non-zero) values that are part of the networkx
    solution by doing the following:

    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> pairs = G.get_two_hop_neighbors()
    >>> df = cugraph.jaccard(G, pairs)

    But please remember that cugraph will fill the dataframe with the entire
    solution you request, so you'll need enough memory to store the 2-hop
    neighborhood dataframe.


    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the jaccard coefficient for all
        adjacent vertices in the graph.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Jaccard weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID (will be identical to first if specified)
        df['destination'] : cudf.Series
            The destination vertex ID (will be identical to second if
            specified)
        df['jaccard_coeff'] : cudf.Series
            The computed Jaccard coefficient between the source and destination
            vertices

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> df = cugraph.jaccard(G)

    """
    if type(input_graph) is not Graph:
        raise TypeError("input graph must a Graph")

    if type(vertex_pair) == cudf.DataFrame:
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)
    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a cudf dataframe")

    df = jaccard_wrapper.jaccard(input_graph, None, vertex_pair)

    if input_graph.renumbered:
        df = input_graph.unrenumber(df, "source")
        df = input_graph.unrenumber(df, "destination")

    return df


def jaccard_coefficient(G, ebunch=None):
    """
    For NetworkX Compatability.  See `jaccard`

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

    ebunch : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the jaccard coefficient for all
        adjacent vertices in the graph.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Jaccard weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID (will be identical to first if specified)
        df['destination'] : cudf.Series
            The destination vertex ID (will be identical to second if
            specified)
        df['jaccard_coeff'] : cudf.Series
            The computed Jaccard coefficient between the source and destination
            vertices

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> df = cugraph.jaccard_coefficient(G)

    """
    vertex_pair = None

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if isNx is True and ebunch is not None:
        vertex_pair = cudf.DataFrame(ebunch)

    df = jaccard(G, vertex_pair)

    if isNx is True:
        df = df_edge_score_to_dictionary(df,
                                         k="jaccard_coeff",
                                         src="source",
                                         dst="destination")

    return df
