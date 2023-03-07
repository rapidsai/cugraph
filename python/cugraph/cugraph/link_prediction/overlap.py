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

from cugraph.link_prediction import overlap_wrapper
import cudf
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_edge_score_to_dictionary,
    renumber_vertex_pair,
)


def overlap_coefficient(G, ebunch=None):
    """
    For NetworkX Compatability.  See `overlap`

    """
    vertex_pair = None

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if isNx is True and ebunch is not None:
        vertex_pair = cudf.DataFrame(ebunch)

    df = overlap(G, vertex_pair)

    if isNx is True:
        df = df_edge_score_to_dictionary(
            df, k="overlap_coeff", src="first", dst="second"
        )

    return df


def overlap(input_graph, vertex_pair=None):
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
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm). The
        adjacency list will be computed if not already present.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the overlap coefficient is computed for the
        given vertex pairs, else, it is computed for all vertex pairs.

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Overlap coefficients. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        df['overlap_coeff'] : cudf.Series
            The computed overlap coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.overlap(G)

    """

    if type(vertex_pair) == cudf.DataFrame:
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)
    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a cudf dataframe")

    df = overlap_wrapper.overlap(input_graph, None, vertex_pair)

    if input_graph.renumbered:
        df = input_graph.unrenumber(df, "first")
        df = input_graph.unrenumber(df, "second")

    return df
