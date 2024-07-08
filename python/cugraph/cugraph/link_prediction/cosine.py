# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
import warnings
from typing import Union, Iterable

from pylibcugraph import (
    cosine_coefficients as pylibcugraph_cosine_coefficients,
    all_pairs_cosine_coefficients as pylibcugraph_all_pairs_cosine_coefficients,
)
from pylibcugraph import ResourceHandle

from cugraph.structure import Graph
from cugraph.utilities.utils import import_optional

# FIXME: the networkx.Graph type used in type annotations is specified
# using a string literal to avoid depending on and importing networkx.
# Instead, networkx is imported optionally, which may cause a problem
# for a type checker if run in an environment where networkx is not installed.
networkx = import_optional("networkx")


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, vertex_pair):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    vertex_pair_dtypes = vertex_pair.dtypes

    if (
        vertex_pair_dtypes.iloc[0] != vertex_dtype
        or vertex_pair_dtypes.iloc[1] != vertex_dtype
    ):
        warning_msg = (
            "Cosine requires 'vertex_pair' to match the graph's 'vertex' type. "
            f"input graph's vertex type is: {vertex_dtype} and got "
            f"'vertex_pair' of type: {vertex_pair_dtypes}."
        )
        warnings.warn(warning_msg, UserWarning)
        vertex_pair = vertex_pair.astype(vertex_dtype)

    return vertex_pair


def cosine(
    input_graph: Graph,
    vertex_pair: cudf.DataFrame = None,
    use_weight: bool = False,
):
    """
    Compute the Cosine similarity between each pair of vertices connected by
    an edge, or between arbitrary pairs of vertices specified by the user.

    cugraph.cosine, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the cosine coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list. The graph should be undirected where an undirected
        edge is represented by a directed edge in both direction.The adjacency
        list will be computed if not already present.

        This implementation only supports undirected, non-multi Graphs.

    vertex_pair : cudf.DataFrame, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices. If provided, the cosine coefficient is computed for the
        given vertex pairs.  If the vertex_pair is not provided then the
        current implementation computes the cosine coefficient for all
        adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted cosine (if use_weight==True)
        or un-weighted cosine (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Cosine weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        df['cosine_coeff'] : cudf.Series
            The computed Cosine coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import cosine
    >>> input_graph = karate.get_graph(download=True, ignore_weights=True)
    >>> df = cosine(input_graph)

    """
    if input_graph.is_directed():
        raise ValueError("Input must be an undirected Graph.")

    if vertex_pair is None:
        # Call two_hop neighbor of the entire graph
        vertex_pair = input_graph.get_two_hop_neighbors()

    v_p_num_col = len(vertex_pair.columns)

    if isinstance(vertex_pair, cudf.DataFrame):
        vertex_pair = renumber_vertex_pair(input_graph, vertex_pair)
        vertex_pair = ensure_valid_dtype(input_graph, vertex_pair)
        src_col_name = vertex_pair.columns[0]
        dst_col_name = vertex_pair.columns[1]
        first = vertex_pair[src_col_name]
        second = vertex_pair[dst_col_name]

    elif vertex_pair is not None:
        raise ValueError("vertex_pair must be a cudf Dataframe")

    first, second, cosine_coeff = pylibcugraph_cosine_coefficients(
        resource_handle=ResourceHandle(),
        graph=input_graph._plc_graph,
        first=first,
        second=second,
        use_weight=use_weight,
        do_expensive_check=False,
    )

    if input_graph.renumbered:
        vertex_pair = input_graph.unrenumber(
            vertex_pair, src_col_name, preserve_order=True
        )
        vertex_pair = input_graph.unrenumber(
            vertex_pair, dst_col_name, preserve_order=True
        )

    if v_p_num_col == 2:
        # single column vertex
        vertex_pair = vertex_pair.rename(
            columns={src_col_name: "first", dst_col_name: "second"}
        )

    df = vertex_pair
    df["cosine_coeff"] = cudf.Series(cosine_coeff)

    return df


def cosine_coefficient(
    G: Union[Graph, "networkx.Graph"],
    ebunch: Union[cudf.DataFrame, Iterable[Union[int, str, float]]] = None,
):
    """
    Note: No NetworkX equivalent.

    Parameters
    ----------
    G : cugraph.Graph or NetworkX.Graph
        cuGraph or NetworkX Graph instance, should contain the connectivity
        information as an edge list. The graph should be undirected where an
        undirected edge is represented by a directed edge in both direction.
        The adjacency list will be computed if not already present.

        This implementation only supports undirected, non-multi Graphs.

    ebunch : cudf.DataFrame or iterable of node pairs, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices or iterable of 2-tuples (u, v) where u and v are nodes in
        the graph.

        If provided, the Overlap coefficient is computed for the given vertex
        pairs. Otherwise, the current implementation computes the overlap
        coefficient for all adjacent vertices in the graph.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Cosine weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            the second vertex ID of each pair (will be identical to second if
            specified).
        df['cosine_coeff'] : cudf.Series
            The computed Cosine coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import cosine_coefficient
    >>> G = karate.get_graph(download=True)
    >>> df = cosine_coefficient(G)

    """
    vertex_pair = None

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if isNx is True and ebunch is not None:
        vertex_pair = cudf.DataFrame(ebunch)

    df = cosine(G, vertex_pair)

    if isNx is True:
        df = df_edge_score_to_dictionary(
            df, k="cosine_coeff", src="first", dst="second"
        )

    return df


def all_pairs_cosine(
    input_graph: Graph,
    vertices: cudf.Series = None,
    use_weight: bool = False,
    topk: int = None,
):
    """
    Compute the All Pairs Cosine similarity between all pairs of vertices specified.
    The Cosine similarity weight of each edge represents the strength of connection
    between vertices based on the relative similarity of their neighbors.

    cugraph.all_pairs_cosine, in the absence of specified vertices, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the cosine coefficient for all the vertex pairs in the graph.
    This is not advisable as the vertex_pairs can grow exponentially with respect to
    the size of the datasets.

    If the topk parameter is specified then the result will only contain the top k
    highest scoring results.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list. The graph should be undirected where an undirected
        edge is represented by a directed edge in both direction.The adjacency
        list will be computed if not already present.

        This implementation only supports undirected, non-multi Graphs.

    vertices : int or list or cudf.Series or cudf.DataFrame, optional (default=None)
        A GPU Series containing the input vertex list.  If the vertex list is not
        provided then the current implementation computes the cosine coefficient for
        all adjacent vertices in the graph.

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted cosine (if use_weight==True)
        or un-weighted cosine (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.

    topk : int, optional (default=None)
        Specify the number of answers to return otherwise returns the entire
        solution

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the Cosine weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        df['first'] : cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        df['second'] : cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        df['cosine_coeff'] : cudf.Series
            The computed Cosine coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import all_pairs_cosine
    >>> input_graph = karate.get_graph(download=True, ignore_weights=True)
    >>> df = all_pairs_cosine(input_graph)

    """
    if input_graph.is_directed():
        raise ValueError("Input must be an undirected Graph.")

    if vertices is not None:

        if isinstance(vertices, int):
            vertices = [vertices]

        if isinstance(vertices, list):
            vertices = cudf.Series(
                vertices,
                dtype=input_graph.edgelist.edgelist_df[input_graph.srcCol].dtype,
            )

        if input_graph.renumbered is True:
            if isinstance(vertices, cudf.DataFrame):
                vertices = input_graph.lookup_internal_vertex_id(
                    vertices, vertices.columns
                )
            else:
                vertices = input_graph.lookup_internal_vertex_id(vertices)

    first, second, cosine_coeff = pylibcugraph_all_pairs_cosine_coefficients(
        resource_handle=ResourceHandle(),
        graph=input_graph._plc_graph,
        vertices=vertices,
        use_weight=use_weight,
        topk=topk,
        do_expensive_check=False,
    )
    vertex_pair = cudf.DataFrame()
    vertex_pair["first"] = first
    vertex_pair["second"] = second

    if input_graph.renumbered:
        vertex_pair = input_graph.unrenumber(vertex_pair, "first", preserve_order=True)
        vertex_pair = input_graph.unrenumber(vertex_pair, "second", preserve_order=True)

    df = vertex_pair
    df["cosine_coeff"] = cudf.Series(cosine_coeff)

    return df
