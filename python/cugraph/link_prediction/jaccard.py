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

from cugraph.link_prediction import jaccard_wrapper
from cugraph.structure.graph import null_check
import cudf


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

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.
    vertex_pair : cudf.DataFrame
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the given
        vertex pairs, else, it is computed for all vertex pairs.  

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
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> df = cugraph.jaccard(G)
    """

    if (type(vertex_pair) == cudf.DataFrame):
        null_check(vertex_pair[vertex_pair.columns[0]])
        null_check(vertex_pair[vertex_pair.columns[1]])
    elif vertex_pair is None:
        pass
    else:
        raise ValueError("vertex_pair must be a cudf dataframe")

    df = jaccard_wrapper.jaccard(input_graph, vertex_pair)

    return df
