# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcugraph import (
    ResourceHandle,
    node2vec_random_walks as pylibcugraph_node2vec_random_walks,
)
import warnings

import cudf


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph, start_vertices):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    if isinstance(start_vertices, cudf.Series):
        start_vertices_dtype = start_vertices.dtype
    else:
        start_vertices_dtype = start_vertices.dtypes.iloc[0]

    if start_vertices_dtype != vertex_dtype:
        warning_msg = (
            "Node2vec requires 'start_vertices' to match the graph's "
            f"'vertex' type. input graph's vertex type is: {vertex_dtype} and got "
            f"'start_vertices' of type: {start_vertices_dtype}."
        )
        warnings.warn(warning_msg, UserWarning)
        start_vertices = start_vertices.astype(vertex_dtype)

    return start_vertices


def node2vec_random_walks(
    G, start_vertices, max_depth=1, p=1.0, q=1.0, random_state=None
):
    """
    Computes random walks for each node in 'start_vertices', under the
    node2vec sampling framework.

    References
    ----------

    A Grover, J Leskovec: node2vec: Scalable Feature Learning for Networks,
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining, https://arxiv.org/abs/1607.00653

    Parameters
    ----------
    G : cuGraph.Graph
        The graph can be either directed or undirected.
        Weights in the graph are ignored.

    start_vertices: int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame. Only supports int32 currently.

    max_depth: int, optional (default=1)
        The maximum depth of the random walks. If not specified, the maximum
        depth is set to 1.

    p: float, optional (default=1.0, [0 < p])
        Return factor, which represents the likelihood of backtracking to
        a previous node in the walk. A higher value makes it less likely to
        sample a previously visited node, while a lower value makes it more
        likely to backtrack, making the walk "local". A positive float.

    q: float, optional (default=1.0, [0 < q])
        In-out factor, which represents the likelihood of visiting nodes
        closer or further from the outgoing node. If q > 1, the random walk
        is likelier to visit nodes closer to the outgoing node. If q < 1, the
        random walk is likelier to visit nodes further from the outgoing node.
        A positive float.

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
    >>> G = karate.get_graph(download=True)
    >>> start_vertices = cudf.Series([0, 2], dtype=np.int32)
    >>> paths, weights, max_length = cugraph.node2vec_random_walks(G,
    ...                                               start_vertices, 3,
    ...                                               0.8, 0.5)

    """
    if (not isinstance(max_depth, int)) or (max_depth < 1):
        raise ValueError(
            f"'max_depth' must be a positive integer, " f"got: {max_depth}"
        )
    if (not isinstance(p, float)) or (p <= 0.0):
        raise ValueError(f"'p' must be a positive float, got: {p}")
    if (not isinstance(q, float)) or (q <= 0.0):
        raise ValueError(f"'q' must be a positive float, got: {q}")

    if isinstance(start_vertices, int):
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices, dtype="int32")
        # FIXME: Verify if this condition still holds
        if start_vertices.dtype != "int32":
            raise ValueError(
                f"'start_vertices' must have int32 values, "
                f"got: {start_vertices.dtype}"
            )

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices, start_vertices.columns
            )
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

    start_vertices = ensure_valid_dtype(G, start_vertices)

    vertex_paths, edge_wgt_paths = pylibcugraph_node2vec_random_walks(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        seed_array=start_vertices,
        max_depth=max_depth,
        p=p,
        q=q,
        random_state=random_state,
    )
    vertex_paths = cudf.Series(vertex_paths)
    edge_wgt_paths = cudf.Series(edge_wgt_paths)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_["vertex_paths"] = vertex_paths
        df_ = G.unrenumber(df_, "vertex_paths", preserve_order=True)
        if len(df_.columns) > 1:
            vertex_paths = df_.fillna(-1)
        else:
            vertex_paths = cudf.Series(df_["vertex_paths"]).fillna(-1)

    return vertex_paths, edge_wgt_paths, max_depth
