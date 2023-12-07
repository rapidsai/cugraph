# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
import cupy as cp
from pylibcugraph import ResourceHandle
from pylibcugraph import (
    uniform_random_walks as pylibcugraph_uniform_random_walks,
)

from cugraph.utilities import ensure_cugraph_obj_for_nx
from cugraph.structure import Graph

import warnings
from cugraph.utilities.utils import import_optional
from typing import Union, Tuple

# FIXME: the networkx.Graph type used in type annotations is specified
# using a string literal to avoid depending on and importing networkx.
# Instead, networkx is imported optionally, which may cause a problem
# for a type checker if run in an environment where networkx is not installed.
networkx = import_optional("networkx")


def uniform_random_walks(
    G: Graph,
    start_vertices: Union[int, list, cudf.Series, cudf.DataFrame] = None,
    max_depth: int = None,
) -> Tuple[cp.ndarray, cp.ndarray, int]:
    return pylibcugraph_uniform_random_walks(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertices=start_vertices,
        max_length=max_depth,
    )


def random_walks(
    G: Union[Graph, "networkx.Graph"],
    random_walks_type: str = "uniform",
    start_vertices: Union[int, list, cudf.Series, cudf.DataFrame] = None,
    max_depth: int = None,
    use_padding: bool = False,
    legacy_result_type: bool = True,
) -> Tuple[cudf.Series, cudf.Series, Union[None, int, cudf.Series]]:
    """
    Compute random walks for each nodes in 'start_vertices' and returns
    either a padded or a coalesced result. For the padded case, vertices
    with no outgoing edges will be padded with -1.

    When 'use_padding' is 'False', 'random_walks' returns a coalesced
    result which is a compressed version of the padded one. In the padded
    form, sources with no out_going edges are padded with -1s in the
    'vertex_paths' array and their corresponding edges('edge_weight_paths')
    with 0.0s (when 'legacy_result_type' is 'True'). If 'legacy_result_type'
    is 'False', 'random_walks' returns padded results (vertex_paths,
    edge_weight_paths) but instead of 'sizes = None', returns the 'max_path_lengths'.
    When 'legacy_result_type' is 'False', the arhument 'use_padding' is ignored.

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

        When 'legacy_result_type' is set to False, 'max_depth' is relative to
        the number of edges otherwised, it is relative to the number of vertices.

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
    sizes: None or cudf.Series
        The path sizes in case of 'coalesced' paths or None if 'padded'.
    or
    max_path_length : int
        The maximum path length if 'legacy_result_type' is 'False'

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> M = karate.get_edgelist(download=True)
    >>> G = karate.get_graph()
    >>> start_vertices = G.nodes()[:4]
    >>> _, _, _ = cugraph.random_walks(G, "uniform", start_vertices, 3)

    """

    if legacy_result_type:
        warning_msg = (
            "Coalesced path results, returned when setting legacy_result_type=True, "
            "is deprecated and will no longer be supported in the next releases. "
            "only padded paths will be returned instead"
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
        # Ensure the 'start_vertices' have the same dtype as the edge list.
        # Failing to do that may produce erroneous results.
        vertex_dtype = G.edgelist.edgelist_df.dtypes[0]
        start_vertices = cudf.Series(start_vertices, dtype=vertex_dtype)

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

    vertex_paths = cudf.Series(vertex_paths)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_["vertex_paths"] = vertex_paths
        df_ = G.unrenumber(df_, "vertex_paths", preserve_order=True)
        vertex_paths = cudf.Series(df_["vertex_paths"]).fillna(-1)

    edge_wgt_paths = cudf.Series(edge_wgt_paths)

    # The PLC uniform random walks returns an extra vertex along with an extra
    # edge per path. In fact, the max depth is relative to the number of vertices
    # for the legacy implementation and edges for the PLC implementation

    if legacy_result_type:
        warning_msg = (
            "The 'max_depth' is relative to the number of vertices and will be "
            "deprecated in the next release. For non legacy result type, it is "
            "relative to the number of edges which will only be supported."
        )
        warnings.warn(warning_msg, PendingDeprecationWarning)

        # Drop the last vertex and and edge weight from each vertex and edge weight
        # paths.
        vertex_paths = vertex_paths.drop(
            index=vertex_paths[max_depth :: max_depth + 1].index
        ).reset_index(drop=True)

        edge_wgt_paths = edge_wgt_paths.drop(
            index=edge_wgt_paths[max_depth - 1 :: max_depth].index
        ).reset_index(drop=True)

        if use_padding:
            sizes = None
            # FIXME: Is it necessary to slice it with 'edge_wgt_paths_sz'?
            return vertex_paths, edge_wgt_paths, sizes

        # If 'use_padding' is False, compute the sizes of the unpadded results

        sizes = (
            vertex_paths.apply(lambda x: 1 if x != -1 else 0)
            .groupby(vertex_paths.index // max_depth, sort=True)
            .sum()
            .reset_index(drop=True)
        )

        # Drop the -1 values which are representative of no outgoing edges
        vertex_paths = vertex_paths.pipe(lambda x: x[x != -1]).reset_index(drop=True)

        # Drop the 0.0 values which are representative of no edges.
        edge_wgt_paths = edge_wgt_paths.pipe(lambda x: x[x != 0.0]).reset_index(
            drop=True
        )

        return vertex_paths, edge_wgt_paths, sizes

    else:
        return (
            vertex_paths,
            edge_wgt_paths,
            max_path_length,
        )


def rw_path(
    num_paths: int, sizes: cudf.Series
) -> Tuple[cudf.Series, cudf.Series, cudf.Series]:
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
