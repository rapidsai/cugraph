# Copyright (c) 2022, NVIDIA CORPORATION.
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

from pylibcugraph import ResourceHandle
from pylibcugraph import uniform_random_walks as pylibcugraph_uniform_random_walks
from cugraph.utilities import ensure_cugraph_obj_for_nx

# FIXME: get rid of this call as it is using 'cython.cu'
from cugraph.sampling import random_walks_wrapper

import cudf


def uniform_random_walks(G, start_vertices, max_depth=None, use_padding=False):
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
    >>> _, _, _, _ = cugraph.random_walks(G, M, 3)

    """
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

    # FIXME: Match the start_vertices type to the edgelist type
    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices, dtype="int32")

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices, start_vertices.columns
            )
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

    sizes = None
    vertex_set, edge_set, max_path_length = pylibcugraph_uniform_random_walks(
        resource_handle=ResourceHandle(),
        input_graph=G._plc_graph,
        start_vertices=start_vertices,
        max_length=max_depth,
    )

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_["vertex_set"] = vertex_set
        df_ = G.unrenumber(df_, "vertex_set", preserve_order=True)
        vertex_set = cudf.Series(df_["vertex_set"])

    # FIXME: The call below is from the legacy implementation
    # What difference does this make?
    if use_padding:
        edge_set_sz = (max_depth - 1) * len(start_vertices)
        return vertex_set, edge_set[:edge_set_sz], sizes

    # FIXME: What is the use of this?
    """
    vertex_set_sz = sizes.sum()
    edge_set_sz = vertex_set_sz - len(start_vertices)
    """
    # FIXME: wouldn't 'vertex_set_sz' and 'edge_set_sz' always be the
    # size of 'vertex_set' and 'edge_set'?
    # return vertex_set[:vertex_set_sz], edge_set[:edge_set_sz], sizes
    return vertex_set, edge_set, max_path_length


def rw_path(num_paths, sizes):
    """
    Retrieve more information on the obtained paths in case use_padding
    is False.

    parameters
    ----------
    num_paths: int
        Number of paths in the random walk output.

    sizes: int
        Path size returned in random walk output.

    Returns
    -------
    path_data : cudf.DataFrame
        Dataframe containing vetex path offsets, edge weight offsets and
        edge weight sizes for each path.
    """
    # FIXME: leverage dask to get these metrics since the call below
    # lives in cython.cu ?
    # Is the below code generic, or does it retrieve/leverage the results
    # from 'random_walks'?
    return random_walks_wrapper.rw_path_retrieval(num_paths, sizes)
