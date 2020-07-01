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

from cugraph.link_prediction import overlap_wrapper
from cugraph.structure.graph import null_check
import cudf


def overlap_w(input_graph, weights, vertex_pair=None):
    """
    Compute the weighted Overlap Coefficient between each pair of vertices
    connected by an edge, or between arbitrary pairs of vertices specified by
    the user. Overlap Coefficient is defined between two sets as the ratio of
    the volume of their intersection divided by the smaller of their volumes.
    In the context of graphs, the neighborhood of a vertex is seen as a set.
    The Overlap Coefficient weight of each edge represents the strength of
    connection between vertices based on the relative similarity of their
    neighbors. If first is specified but second is not, or vice versa, an
    exception will be thrown.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        adjacency list will be computed if not already present.

    weights : cudf.Series
        Specifies the weights to be used for each vertex.

    vertex_pair : cudf.DataFrame
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the overlap coefficient is computed for the
        given vertex pairs, else, it is computed for all vertex pairs.

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the overlap coefficients. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['source'] : cudf.Series
            The source vertex ID
        df['destination'] : cudf.Series
            The destination vertex ID
        df['overlap_coeff'] : cudf.Series
            The computed weighted Overlap coefficient between the source and
            destination vertices.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> df = cugraph.overlap_w(G, M[2])
    """

    renumber_map = None

    if input_graph.renumbered:
        renumber_map = input_graph.edgelist.renumber_map

    if type(vertex_pair) == cudf.DataFrame:
        for col in vertex_pair.columns:
            null_check(vertex_pair[col])
            if input_graph.renumbered:
                vertex_pair = renumber_map.add_vertex_id(
                    vertex_pair, "id", col, drop=True
                ).rename({"id": col})
    elif vertex_pair is None:
        pass
    else:
        raise ValueError("vertex_pair must be a cudf dataframe")

    df = overlap_wrapper.overlap(input_graph, weights, vertex_pair)

    if input_graph.renumbered:
        # FIXME: multi column support
        df = renumber_map.from_vertex_id(df, "source", drop=True).rename(
            {"0": "source"}
        )
        df = renumber_map.from_vertex_id(df, "destination", drop=True).rename(
            {"0": "destination"}
        )

    return df
