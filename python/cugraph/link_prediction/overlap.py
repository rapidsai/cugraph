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

from cugraph.link_prediction import overlap_wrapper
from cugraph.structure.graph import null_check
import cudf


def overlap(input_graph, first=None, second=None):
    """
    Compute the Overlap Coefficient between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.
    Overlap Coefficient is defined between two sets as the ratio of the volume
    of their intersection divided by the smaller of their two volumes. In the
    context of graphs, the neighborhood of a vertex is seen as a set. The
    Overlap Coefficient weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors. If first is specified but second is not, or vice versa, an
    exception will be thrown.

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        adjacency list will be computed if not already present.
    first : cudf.Series
        Specifies the first vertices of each pair of vertices to compute for,
        must be specified along with second.
    second : cudf.Series
        Specifies the second vertices of each pair of vertices to compute for,
        must be specified along with first.

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Overlap coefficients. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID (will be identical to first if specified).
        df['destination'] : cudf.Series
            The destination vertex ID (will be identical to second if
            specified).
        df['overlap_coeff'] : cudf.Series
            The computed Overlap coefficient between the source and destination
            vertices.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> df = cugraph.overlap(G)
    """

    if (type(first) == cudf.Series and
            type(second) == cudf.Series):
        null_check(first)
        null_check(second)
    elif first is None and second is None:
        pass
    else:
        raise ValueError("Specify first and second or neither")

    df = overlap_wrapper.overlap(input_graph, first, second)

    return df
