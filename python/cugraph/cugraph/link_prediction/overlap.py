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
    overlap_coefficients as pylibcugraph_overlap_coefficients,
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
            "Overlap requires 'vertex_pair' to match the graph's 'vertex' type. "
            f"input graph's vertex type is: {vertex_dtype} and got "
            f"'vertex_pair' of type: {vertex_pair_dtypes}."
        )
        warnings.warn(warning_msg, UserWarning)
        vertex_pair = vertex_pair.astype(vertex_dtype)

    return vertex_pair


def overlap_coefficient(
    G: Union[Graph, "networkx.Graph"],
    ebunch: Union[cudf.DataFrame, Iterable[Union[int, str, float]]] = None,
    do_expensive_check: bool = False,  # deprecated
):
    """
    Compute overlap coefficient.

    Parameters
    ----------
    G : cugraph.Graph or NetworkX.Graph
        cuGraph or NetworkX Graph instance, should contain the connectivity
        information as an edge list. The graph should be undirected where an
        undirected edge is represented by a directed edge in both direction.
        The adjacency list will be computed if not already present.

        This implementation only supports undirected, non-multi edge Graph.

        .. deprecated:: 24.08

           Accepting ``networkx.Graph`` is deprecated and will be removed in a
           future version.  For ``networkx.Graph`` use netwotkx directly with
           the ``nx-cugraph`` plug-in.

    ebunch : cudf.DataFrame or iterable of node pairs, optional (default=None)
        A GPU dataframe consisting of two columns representing pairs of
        vertices or iterable of 2-tuples (u, v) where u and v are nodes in
        the graph.

        If provided, the Overlap coefficient is computed for the given vertex
        pairs. Otherwise, the current implementation computes the overlap
        coefficient for all adjacent vertices in the graph.

    do_expensive_check : bool, optional (default=False)
        Deprecated.
        This option added a check to ensure integer vertex IDs are sequential
        values from 0 to V-1. That check is now redundant because cugraph
        unconditionally renumbers and un-renumbers integer vertex IDs for
        optimal performance, therefore this option is deprecated and will be
        removed in a future version.

    Returns
    -------
    df  : cudf.DataFrame
        GPU data frame of size E (the default) or the size of the given pairs
        (first, second) containing the overlap weights. The ordering is
        relative to the adjacency list, or that given by the specified vertex
        pairs.

        ddf['first']: dask_cudf.Series
            The first vertex ID of each pair (will be identical to first if specified).
        ddf['second']: dask_cudf.Series
            The second vertex ID of each pair (will be identical to second if
            specified).
        ddf['overlap_coeff']: dask_cudf.Series
            The computed overlap coefficient between the first and the second
            vertex ID.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> from cugraph import overlap_coefficient
    >>> G = karate.get_graph(download=True, ignore_weights=True)
    >>> df = overlap_coefficient(G)
    """
    if do_expensive_check:
        warnings.warn(
            "do_expensive_check is deprecated since vertex IDs are no longer "
            "required to be consecutively numbered",
            FutureWarning,
        )

    vertex_pair = None

    G, isNx = ensure_cugraph_obj_for_nx(G)

    # FIXME: What is the logic behind this since the docstrings mention that 'G' and
    # 'ebunch'(if not None) are respectively of type cugraph.Graph and cudf.DataFrame?
    if isNx is True and ebunch is not None:
        vertex_pair = cudf.DataFrame(ebunch)

    df = overlap(G, vertex_pair)

    if isNx is True:
        df = df_edge_score_to_dictionary(
            df, k="overlap_coeff", src="first", dst="second"
        )

    return df


def overlap(
    input_graph: Graph,
    vertex_pair: cudf.DataFrame = None,
    do_expensive_check: bool = False,  # deprecated
    use_weight: bool = False,
):
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

    cugraph.overlap, in the absence of a specified vertex pair list, will
    compute the two_hop_neighbors of the entire graph to construct a vertex pair
    list and will return the overlap coefficient for those vertex pairs. This is
    not advisable as the vertex_pairs can grow exponentially with respect to the
    size of the datasets

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph Graph instance, should contain the connectivity information
        as an edge list. The adjacency list will be computed if not already
        present.

        This implementation only supports undirected, non-multi edge Graph.
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

    use_weight : bool, optional (default=False)
        Flag to indicate whether to compute weighted overlap (if use_weight==True)
        or un-weighted overlap (if use_weight==False).
        'input_graph' must be weighted if 'use_weight=True'.



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
    >>> from cugraph.datasets import karate
    >>> from cugraph import overlap
    >>> input_graph = karate.get_graph(download=True, ignore_weights=True)
    >>> df = overlap(input_graph)

    """
    if do_expensive_check:
        warnings.warn(
            "do_expensive_check is deprecated since vertex IDs are no longer "
            "required to be consecutively numbered",
            FutureWarning,
        )

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
        raise ValueError("vertex_pair must be a cudf dataframe")

    first, second, overlap_coeff = pylibcugraph_overlap_coefficients(
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
    df["overlap_coeff"] = cudf.Series(overlap_coeff)

    return df
