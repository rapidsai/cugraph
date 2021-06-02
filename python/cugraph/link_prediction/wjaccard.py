# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cugraph.structure.graph_classes import Graph
from cugraph.link_prediction import jaccard_wrapper
import cudf
import numpy as np
from cugraph.utilities import renumber_vertex_pair


def jaccard_w(input_graph, weights, vertex_pair=None):
    """
    Compute the weighted Jaccard similarity between each pair of vertices
    connected by an edge, or between arbitrary pairs of vertices specified by
    the user. Jaccard similarity is defined between two sets as the ratio of
    the volume of their intersection divided by the volume of their union. In
    the context of graphs, the neighborhood of a vertex is seen as a set. The
    Jaccard similarity weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors. If first is specified but second is not, or vice versa, an
    exception will be thrown.

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        adjacency list will be computed if not already present.

    weights : cudf.DataFrame
        Specifies the weights to be used for each vertex.
        Vertex should be represented by multiple columns for multi-column
        vertices.

        weights['vertex'] : cudf.Series
            Contains the vertex identifiers
        weights['weight'] : cudf.Series
            Contains the weights of vertices

    vertex_pair : cudf.DataFrame
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the
        given vertex pairs, else, it is computed for all vertex pairs.

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Jaccard weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID
        df['destination'] : cudf.Series
            The destination vertex ID
        df['jaccard_coeff'] : cudf.Series
            The computed weighted Jaccard coefficient between the source and
            destination vertices.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.jaccard_w(G, M[2])
    """
    if type(input_graph) is not Graph:
        raise Exception("input graph must be undirected")

    if type(vertex_pair) == cudf.DataFrame:
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)
    elif vertex_pair is None:
        pass
    else:
        raise ValueError("vertex_pair must be a cudf dataframe")

    if input_graph.renumbered:
        vertex_size = input_graph.vertex_column_size()
        if vertex_size == 1:
            weights = input_graph.add_internal_vertex_id(
                weights, 'vertex', 'vertex'
            )
        else:
            cols = weights.columns[:vertex_size].to_list()
            weights = input_graph.add_internal_vertex_id(
                weights, 'vertex', cols
            )
    jaccard_weights = cudf.Series(np.ones(len(weights)))
    for i in range(len(weights)):
        jaccard_weights[weights['vertex'].iloc[i]] = weights['weight'].iloc[i]
    df = jaccard_wrapper.jaccard(input_graph, jaccard_weights, vertex_pair)

    if input_graph.renumbered:
        df = input_graph.unrenumber(df, "source")
        df = input_graph.unrenumber(df, "destination")

    return df
