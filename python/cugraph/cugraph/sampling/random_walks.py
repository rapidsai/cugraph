import cudf
from cugraph.experimental.sampling.uniform_random_walks import uniform_random_walks

import warnings


def random_walks(G, start_vertices, max_depth=None, use_padding=False):
    """
    compute random walks for each nodes in 'start_vertices'

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed or undirected.

    start_vertices : int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame

    max_depth : int
        The maximum depth of the random walks

    use_padding : bool, optional (default=False)
        If True, padded paths are returned else coalesced paths are returned.

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    sizes: int
        The path size in case of coalesced paths.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> M = karate.get_edgelist(fetch=True)
    >>> G = karate.get_graph()
    >>> _, _, _ = cugraph.random_walks(G, M, 3)

    """
    warning_msg = (
        "This call is deprecated and will be refactored "
        "in the next release: use 'uniform_random_walks' instead"
    )
    warnings.warn(warning_msg, PendingDeprecationWarning)

    vertex_set, edge_set, _ = uniform_random_walks(G, start_vertices, max_depth)

    # The PLC uniform random walks returns an extra vertex along with an extra
    # edge per path. In fact, the max depth is relative to the number of vertices
    # for the legacy implementation and edges for the PLC implementation

    # Get a list of extra vertex and edge index to drop
    drop_vertex = [i for i in range(max_depth, len(vertex_set), max_depth + 1)]
    drop_edge_wgt = [i - 1 for i in range(max_depth, len(edge_set), max_depth)]

    vertex_set = vertex_set.drop(vertex_set.index[drop_vertex]).reset_index(drop=True)

    edge_set = edge_set.drop(edge_set.index[drop_edge_wgt]).reset_index(drop=True)

    if use_padding:
        sizes = None
        edge_set_sz = (max_depth - 1) * len(start_vertices)
        return vertex_set, edge_set[:edge_set_sz], sizes

    # If 'use_padding' is False, compute the sizes of the unpadded results
    sizes = [
        len(vertex_set.iloc[i : i + max_depth].dropna())
        for i in range(0, len(vertex_set), max_depth)
    ]
    sizes = cudf.Series(sizes, dtype=vertex_set.dtype)

    # Compress the 'vertex_set' by dropping 'NA' values which is representative
    # of vertices with no outgoing link
    vertex_set = vertex_set.dropna().reset_index(drop=True)
    # Compress the 'edge_set' by dropping 'NA'
    edge_set.replace(0.0, None, inplace=True)
    edge_set = edge_set.dropna().reset_index(drop=True)
    return vertex_set, edge_set, sizes


def rw_path(num_paths, sizes):
    """
    Retrieve more information on the obtained paths in case use_padding
    is False.

    parameters
    ----------
    num_paths: int
        Number of paths in the random walk output.

    sizes: cudf.Series
        Path size returned in random walk output.

    Returns
    -------
    path_data : cudf.DataFrame
        Dataframe containing vetex path offsets, edge weight offsets and
        edge weight sizes for each path.
    """

    vertex_offsets = cudf.Series(0, dtype=sizes.dtype)
    vertex_offsets = cudf.concat(
        [vertex_offsets, sizes.cumsum()[:-1]], ignore_index=True
    )
    weight_sizes = sizes - 1

    weight_offsets = cudf.Series(0, dtype=sizes.dtype)
    num_edges = vertex_offsets.diff()[1:] - 1

    weight_offsets = cudf.concat(
        [weight_offsets, num_edges.cumsum()], ignore_index=True
    )
    # FIXME: CUDF bug. concatenating 2 series of type int32 but get a CUDF of type in64
    # have to cast the results
    weight_offsets = weight_offsets.astype(sizes.dtype)

    path_data = cudf.DataFrame()
    path_data["vertex_offsets"] = vertex_offsets
    path_data["weight_sizes"] = weight_sizes
    path_data["weight_offsets"] = weight_offsets

    return path_data

    # return random_walks_wrapper.rw_path_retrieval(num_paths, sizes)
