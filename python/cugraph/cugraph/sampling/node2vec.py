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

from pylibcugraph.experimental import (ResourceHandle,
                                       GraphProperties,
                                       SGGraph,
                                       node2vec as pylibcugraph_node2vec,
                                       )
from cugraph.utilities import ensure_cugraph_obj_for_nx

import cudf


def node2vec(G,
             start_vertices,
             max_depth=None,
             compress_result=True,
             p=1.0,
             q=1.0):
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
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.

    start_vertices: int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame. Only supports int32 currently.

    max_depth: int
        The maximum depth of the random walks

    compress_result: bool, optional (default=True)
        If True, coalesced paths are returned with a sizes array with offsets.
        Otherwise padded paths are returned with an empty sizes array.

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

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    sizes: int or cudf.Series
        The path size or sizes in case of coalesced paths.

    Example
    -------
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1', edge_attr='2')
    >>> start_vertices = cudf.Series([0, 2], dtype=np.int32)
    >>> paths, weights, path_sizes = cugraph.node2vec(G, start_vertices, 3,
    ...                                               True, 0.8, 0.5)

    """
    if (not isinstance(max_depth, int)) or (max_depth < 1):
        raise ValueError(f"'max_depth' must be a positive integer, \
                        got: {max_depth}")
    if (not isinstance(compress_result, bool)):
        raise ValueError(f"'compress_result' must be a bool, \
                        got: {compress_result}")
    if (not isinstance(p, float)) or (p <= 0.0):
        raise ValueError(f"'p' must be a positive float, got: {p}")
    if (not isinstance(q, float)) or (q <= 0.0):
        raise ValueError(f"'q' must be a positive float, got: {q}")

    G, _ = ensure_cugraph_obj_for_nx(G)

    if isinstance(start_vertices, int):
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices)
        if start_vertices.dtype != 'int32':
            raise ValueError(f"'start_vertices' must have int32 values, \
                            got: {start_vertices.dtype}")

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices, start_vertices.columns)
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    weights = G.edgelist.edgelist_df['weights']

    if srcs.dtype != 'int32':
        raise ValueError(f"Graph vertices must have int32 values, \
                        got: {srcs.dtype}")

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_multigraph=G.is_multigraph())
    store_transposed = False
    renumber = False
    do_expensive_check = False

    sg = SGGraph(resource_handle, graph_props, srcs, dsts, weights,
                 store_transposed, renumber, do_expensive_check)

    vertex_set, edge_set, sizes = \
        pylibcugraph_node2vec(resource_handle, sg, start_vertices,
                              max_depth, compress_result, p, q)

    vertex_set = cudf.Series(vertex_set)
    edge_set = cudf.Series(edge_set)
    sizes = cudf.Series(sizes)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_['vertex_set'] = vertex_set
        df_ = G.unrenumber(df_, 'vertex_set', preserve_order=True)
        vertex_set = cudf.Series(df_['vertex_set'])
    return vertex_set, edge_set, sizes
