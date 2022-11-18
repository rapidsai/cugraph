# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

import cudf
from pylibcugraph import ResourceHandle
from pylibcugraph import (
    uniform_random_walks as pylibcugraph_uniform_random_walks,
)

from cugraph.utilities import ensure_cugraph_obj_for_nx

import warnings


def uniform_random_walks(G, start_vertices, max_depth):
    return pylibcugraph_uniform_random_walks(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertices=start_vertices,
        max_length=max_depth,
    )


def random_walks(
    G,
    random_walks_type="uniform",
    start_vertices=None,
    max_depth=None,
    use_padding=False,
    legacy_result_type=True,
):
    """
    # FIXME: make the padded value for vertices with outgoing edges
    # consistent in both SG and MG implementation.
    compute random walks for each nodes in 'start_vertices' and returns
    either a padded or a coalesced result. For the padded case, vertices
    with no outgoing edges will be padded with NA

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed or undirected.

    random_walks_type : str, optional (default='uniform')
        Type of random walks: 'uniform', 'biased', 'node2vec'.
        Only 'uniform' random walks is currently supported

    start_vertices : int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame

    max_depth : int
        The maximum depth of the random walks

    use_padding : bool, optional (default=False)
        If True, padded paths are returned else coalesced paths are returned.

    legacy_result_type : bool, optional (default=True)
        If True, will return a tuple of vertex_paths, edge_weight_paths and
        sizes. If False, will return a tuple of vertex_paths, vertex_paths and
        max_path_length

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    and
    sizes: int
        The path size in case of coalesced paths.
    or
    max_path_length : int
        The maximum path length

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> M = karate.get_edgelist(fetch=True)
    >>> G = karate.get_graph()
    >>> _, _, _ = cugraph.random_walks(G, "uniform", M, 3)

    """
    if legacy_result_type:
        warning_msg = (
            "Coalesced path results is deprecated and will no longer be "
            "supported in the next releases. only padded paths will be "
            "returned instead"
        )
    warnings.warn(warning_msg, PendingDeprecationWarning)

    if max_depth is None:
        raise TypeError("must specify a 'max_depth'")

    # FIXME: supporting Nx types should mean having a return type that better
    # matches Nx expectations (eg. data on the CPU, possibly using a different
    # data struct like a dictionary, etc.). The 2nd value is ignored here,
    # which is typically named isNx and used to convert the return type.
    # Consider a different return type if Nx types are passed in.
    G, _ = ensure_cugraph_obj_for_nx(G)

    if isinstance(start_vertices, int):
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices)

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices, start_vertices.columns
            )
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

    if random_walks_type == "uniform":
        vertex_paths, edge_wgt_paths, max_path_length = uniform_random_walks(
            G, start_vertices, max_depth
        )

    else:
        raise ValueError("Only 'uniform' random walks is currently supported")

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_["vertex_paths"] = vertex_paths
        df_ = G.unrenumber(df_, "vertex_paths", preserve_order=True)
        vertex_paths = cudf.Series(df_["vertex_paths"]).fillna(-1)

    edge_wgt_paths = cudf.Series(edge_wgt_paths)

    # FIXME: Also add a warning here saying that the lesser path will
    # be no longer be supported
    # The PLC uniform random walks returns an extra vertex along with an extra
    # edge per path. In fact, the max depth is relative to the number of vertices
    # for the legacy implementation and edges for the PLC implementation

    # Get a list of extra vertex and edge index to drop
    if legacy_result_type:
        warning_msg = (
            "The 'max_depth' is relative to the number of vertices and will be "
            "deprecated in the next release. For non legacy result type, it is "
            "relative to the number of edges which will only be supported."
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

        drop_vertex = [i for i in range(max_depth, len(vertex_paths), max_depth + 1)]
        drop_edge_wgt = [
            i - 1 for i in range(max_depth, len(edge_wgt_paths), max_depth)
        ]

        vertex_paths = vertex_paths.drop(vertex_paths.index[drop_vertex]).reset_index(
            drop=True
        )

        edge_wgt_paths = edge_wgt_paths.drop(
            edge_wgt_paths.index[drop_edge_wgt]
        ).reset_index(drop=True)

        if use_padding:
            sizes = None
            edge_wgt_paths_sz = (max_depth - 1) * len(start_vertices)
            # FIXME: Is it necessary to bound the 'edge_wgt_paths'?
            return vertex_paths, edge_wgt_paths[:edge_wgt_paths_sz], sizes

        # If 'use_padding' is False, compute the sizes of the unpadded results
        sizes = [
            len(vertex_paths.iloc[i : i + max_depth].dropna())
            for i in range(0, len(vertex_paths), max_depth)
        ]
        sizes = cudf.Series(sizes, dtype=vertex_paths.dtype)

        # Compress the 'vertex_paths' by dropping 'NA' values which is
        # representative of vertices with no outgoing link
        vertex_paths = vertex_paths.dropna().reset_index(drop=True)
        # Compress the 'edge_wgt_paths' by dropping 'NA'
        edge_wgt_paths.replace(0.0, None, inplace=True)
        edge_wgt_paths = edge_wgt_paths.dropna().reset_index(drop=True)

        vertex_paths_sz = sizes.sum()
        edge_wgt_paths_sz = vertex_paths_sz - len(start_vertices)
        # FIXME: Is it necessary to bound the 'vertex_paths' and 'edge_wgt_paths'?
        return vertex_paths[:vertex_paths_sz], edge_wgt_paths[:edge_wgt_paths_sz], sizes

    else:
        vertex_paths_sz = sizes.sum()
        edge_wgt_paths_sz = vertex_paths_sz - len(start_vertices)
        # FIXME: Is it necessary to bound the 'vertex_paths' and 'edge_wgt_paths'?
        return (
            vertex_paths[:vertex_paths_sz],
            edge_wgt_paths[:edge_wgt_paths_sz],
            max_path_length,
        )


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
    # FIXME: CUDF bug. concatenating two series of type int32 but get a CUDF of
    # type 'int64' have to cast the results
    weight_offsets = weight_offsets.astype(sizes.dtype)

    path_data = cudf.DataFrame()
    path_data["vertex_offsets"] = vertex_offsets
    path_data["weight_sizes"] = weight_sizes
    path_data["weight_offsets"] = weight_offsets

    return path_data[:num_paths]
