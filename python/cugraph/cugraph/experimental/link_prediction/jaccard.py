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

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_edge_score_to_dictionary,
    renumber_vertex_pair,
)
import cudf

from pylibcugraph.experimental import (
    jaccard_coefficients as pylibcugraph_jaccard_coefficients,
)
from pylibcugraph import ResourceHandle


def EXPERIMENTAL__jaccard(G, vertex_pair=None, use_weight=False):
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
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the jaccard coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets

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

    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True, ignore_weights=True)
    >>> pairs = G.get_two_hop_neighbors()
    >>> df = cugraph.jaccard(G, pairs)

    But please remember that cugraph will fill the dataframe with the entire
    solution you request, so you'll need enough memory to store the 2-hop
    neighborhood dataframe.


    Parameters
    ----------
    G : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not supported yet for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

        This implementation only supports undirected, unweighted Graph.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the jaccard coefficient for all
        adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Currently not supported

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Jaccard weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        df['jaccard_coeff'] : cudf.Series
            The computed jaccard coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> from cugraph.experimental import jaccard as exp_jaccard
    >>> G = karate.get_graph(fetch=True, ignore_weights=True)
    >>> df = exp_jaccard(G)

    """
    if G.is_directed():
        raise ValueError("Input must be an undirected Graph.")

    if G.is_weighted():
        raise ValueError("Weighted graphs are currently not supported.")

    if use_weight:
        raise ValueError("'use_weight' is currently not supported.")

    if vertex_pair is None:
        # Call two_hop neighbor of the entire graph
        vertex_pair = G.get_two_hop_neighbors()

    v_p_num_col = len(vertex_pair.columns)

    if isinstance(vertex_pair, cudf.DataFrame):
        vertex_pair = renumber_vertex_pair(G, vertex_pair)
        src_col_name = vertex_pair.columns[0]
        dst_col_name = vertex_pair.columns[1]
        first = vertex_pair[src_col_name]
        second = vertex_pair[dst_col_name]

    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a cudf dataframe")

    use_weight = False
    first, second, jaccard_coeff = pylibcugraph_jaccard_coefficients(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=False,
    )

    if G.renumbered:
        vertex_pair = G.unrenumber(vertex_pair, src_col_name, preserve_order=True)
        vertex_pair = G.unrenumber(vertex_pair, dst_col_name, preserve_order=True)

    if v_p_num_col == 2:
        # single column vertex
        vertex_pair = vertex_pair.rename(
            columns={src_col_name: "first", dst_col_name: "second"}
        )

    df = vertex_pair
    df["jaccard_coeff"] = cudf.Series(jaccard_coeff)

    return df


def EXPERIMENTAL__jaccard_coefficient(G, ebunch=None, use_weight=False):
    """
    For NetworkX Compatability.  See `jaccard`

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list (edge weights are not supported yet for this algorithm). The
        graph should be undirected where an undirected edge is represented by a
        directed edge in both direction. The adjacency list will be computed if
        not already present.

    ebunch : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the jaccard coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the jaccard coefficient for all
        adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Currently not supported

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Jaccard weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        ddf['jaccard_coeff']: dask_cudf.Series
            The computed jaccard coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> from cugraph.experimental import jaccard_coefficient as exp_jaccard_coefficient
    >>> G = karate.get_graph(fetch=True, ignore_weights=True)
    >>> df = exp_jaccard_coefficient(G)

    """
    vertex_pair = None

    G, isNx = ensure_cugraph_obj_for_nx(G)

    # FIXME: What is the logic behind this since the docstrings mention that 'G' and
    # 'ebunch'(if not None) are respectively of type cugraph.Graph and cudf.DataFrame?
    if isNx is True and ebunch is not None:
        vertex_pair = cudf.DataFrame(ebunch)

    df = EXPERIMENTAL__jaccard(G, vertex_pair)

    if isNx is True:
        df = df_edge_score_to_dictionary(
            df, k="jaccard_coeff", src="first", dst="second"
        )

    return df
