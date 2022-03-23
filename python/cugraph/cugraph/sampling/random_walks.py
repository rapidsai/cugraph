# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
from cugraph.sampling import random_walks_wrapper
from cugraph.utilities import ensure_cugraph_obj_for_nx


def random_walks(G,
                 start_vertices,
                 max_depth=None,
                 use_padding=False):
    """
    compute random walks for each nodes in 'start_vertices'

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

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
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2')
    >>> _, _, _ = cugraph.random_walks(G, M, 3)

    """
    if max_depth is None:
        raise TypeError("must specify a 'max_depth'")

    # FIXME: supporting Nx types should mean having a return type that better
    # matches Nx expectations (eg. data on the CPU, possibly using a different
    # data struct like a dictionary, etc.). The 2nd value is ignored here,
    # which is typically named isNx and used to convert the return type.
    # Consider a different return type if Nx types are passed in.
    G, _ = ensure_cugraph_obj_for_nx(G)

    if start_vertices is int:
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices)

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices,
                start_vertices.columns)
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

    vertex_set, edge_set, sizes = random_walks_wrapper.random_walks(
        G, start_vertices, max_depth, use_padding)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_['vertex_set'] = vertex_set
        df_ = G.unrenumber(df_, 'vertex_set', preserve_order=True)
        vertex_set = cudf.Series(df_['vertex_set'])

    if use_padding:
        edge_set_sz = (max_depth-1)*len(start_vertices)
        return vertex_set, edge_set[:edge_set_sz], sizes

    vertex_set_sz = sizes.sum()
    edge_set_sz = vertex_set_sz - len(start_vertices)
    return vertex_set[:vertex_set_sz], edge_set[:edge_set_sz], sizes


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
    return random_walks_wrapper.rw_path_retrieval(num_paths, sizes)
