# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
from pylibcugraph import ResourceHandle
from pylibcugraph import (
    biased_random_walks as pylibcugraph_biased_random_walks,
)

from cugraph.structure import Graph

from typing import Union, Tuple


def biased_random_walks(
    G: Graph,
    start_vertices: Union[int, list, cudf.Series, cudf.DataFrame] = None,
    max_depth: int = None,
    random_state: int = None,
) -> Tuple[cudf.Series, cudf.Series, Union[None, int, cudf.Series]]:
    """
    Compute biased random walks for each nodes in 'start_vertices'.
    Vertices with no outgoing edges will be padded with -1 and the corresponding
    edge weights with 0.0.

    parameters
    ----------
    G : cuGraph.Graph
        The graph can be either directed or undirected.

    start_vertices : int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame

    max_depth : int
        The maximum depth of the random walks

        The max depth is relative to the number of edges hence the vertex_paths size
        is max_depth + 1. For instance, a 'max_depth' of 2 with only one seed will
        result in a vertex_path of size 3.

    random_state: int, optional
        Random seed to use when making sampling calls.


    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    and

    max_path_length : int
        The maximum path length.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> M = karate.get_edgelist(download=True)
    >>> G = karate.get_graph()
    >>> start_vertices = [0, 1]
    >>> paths, weights, max_length = cugraph.biased_random_walks(
    ...            G, start_vertices, 3, random_state=2)

    >>> paths.to_cupy()
    array([ 0,  2,  8, 32,  1, 17,  0,  2], dtype=int32)
    >>> weights.to_cupy()
    array([1., 1., 1., 1., 1., 1.], dtype=float32)
    >>> max_length
    3

    """

    if max_depth is None:
        raise TypeError("must specify a 'max_depth'")

    if isinstance(start_vertices, int):
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        # Ensure the 'start_vertices' have the same dtype as the edge list.
        # Failing to do that may produce erroneous results.
        vertex_dtype = G.edgelist.edgelist_df.dtypes.iloc[0]
        start_vertices = cudf.Series(start_vertices, dtype=vertex_dtype)

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices, start_vertices.columns
            )
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

    vertex_paths, edge_wgt_paths, max_path_length = pylibcugraph_biased_random_walks(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertices=start_vertices,
        max_length=max_depth,
        random_state=random_state,
    )

    vertex_paths = cudf.Series(vertex_paths)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_["vertex_paths"] = vertex_paths
        df_ = G.unrenumber(df_, "vertex_paths", preserve_order=True)
        if len(df_.columns) > 1:
            vertex_paths = df_.fillna(-1)
        else:
            vertex_paths = cudf.Series(df_["vertex_paths"]).fillna(-1)

    edge_wgt_paths = cudf.Series(edge_wgt_paths)

    return (
        vertex_paths,
        edge_wgt_paths,
        max_path_length,
    )
