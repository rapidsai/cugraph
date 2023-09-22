# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from cugraph.link_prediction import overlap
import cudf
import warnings

from cugraph.structure import Graph
from cugraph.utilities.utils import import_optional

# FIXME: the networkx.Graph type used in type annotations is specified
# using a string literal to avoid depending on and importing networkx.
# Instead, networkx is imported optionally, which may cause a problem
# for a type checker if run in an environment where networkx is not installed.
networkx = import_optional("networkx")


def overlap_w(
    input_graph: Graph,
    weights: cudf.DataFrame = None,  # deprecated
    vertex_pair: cudf.DataFrame = None,
    do_expensive_check: bool = False,  # deprecated
):
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

    NOTE: This algorithm doesn't currently support datasets with vertices that
    are not (re)numebred vertices from 0 to V-1 where V is the total number of
    vertices as this creates isolated vertices.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
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

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the overlap coefficient is computed for the
        given vertex pairs, else, it is computed for all vertex pairs.

    do_expensive_check : bool, optional (default=False)
        Deprecated.
        This option added a check to ensure integer vertex IDs are sequential
        values from 0 to V-1. That check is now redundant because cugraph
        unconditionally renumbers and un-renumbers integer vertex IDs for
        optimal performance, therefore this option is deprecated and will be
        removed in a future version.

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the overlap coefficients. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

         df['first'] : cudf.Series
            The first vertex ID of each pair.

        df['second'] : cudf.Series
            The second vertex ID of each pair.

        df['overlap_coeff'] : cudf.Series
            The computed weighted Overlap coefficient between the first and the
            second vertex ID.

    Examples
    --------
    >>> import random
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> # Create a dataframe containing the vertices with their
    >>> # corresponding weight
    >>> weights = cudf.DataFrame()
    >>> # Sample 10 random vertices from the graph and drop duplicates if
    >>> # there are any to avoid duplicates vertices with different weight
    >>> # value in the 'weights' dataframe
    >>> weights['vertex'] = G.nodes().sample(n=10).drop_duplicates()
    >>> # Reset the indices and drop the index column
    >>> weights.reset_index(inplace=True, drop=True)
    >>> # Create a weight column with random weights
    >>> weights['weight'] = [random.random() for w in range(
    ...                      len(weights['vertex']))]
    >>> df = cugraph.overlap_w(G, weights)
    """
    warning_msg = (
        " overlap_w is deprecated. To compute weighted overlap, please use "
        "overlap(input_graph, vertex_pair=False, use_weight=True)"
    )
    warnings.warn(warning_msg, FutureWarning)
    return overlap(input_graph, vertex_pair, do_expensive_check, use_weight=True)
