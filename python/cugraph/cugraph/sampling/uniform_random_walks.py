# Copyright (c) 2024, NVIDIA CORPORATION.
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

from cugraph.structure import Graph

import warnings
from typing import Union, Tuple


def uniform_random_walks(
    G: Graph,
    start_vertices: Union[int, list, cudf.Series, cudf.DataFrame] = None,
    max_depth: int = None,
) -> Tuple[cudf.Series, cudf.Series, Union[None, int, cudf.Series]]:
    """
    Compute uniform random walks for each nodes in 'start_vertices'.
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
    >>> start_vertices = G.nodes()[:4]
    >>> _, _, _ = cugraph.uniform_random_walks(G, start_vertices, 3)

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

    vertex_paths, edge_wgt_paths, max_path_length = pylibcugraph_uniform_random_walks(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertices=start_vertices,
        max_length=max_depth,
    )

    vertex_paths = cudf.Series(vertex_paths)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_["vertex_paths"] = vertex_paths
        df_ = G.unrenumber(df_, "vertex_paths", preserve_order=True)
        vertex_paths = cudf.Series(df_["vertex_paths"]).fillna(-1)

    edge_wgt_paths = cudf.Series(edge_wgt_paths)

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
        Dataframe containing vertex path offsets, edge weight offsets and
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
