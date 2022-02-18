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

import pylibcugraph
import cupy
# import numpy, cudf


def node2vec(G, start_vertices, max_depth, use_padding, p=1.0, q=1.0):
    """
    Computes random walks for each node in 'start_vertices', under the
    node2vec sampling framework described in:

    A Grover, J Leskovec: node2vec: Scalable Feature Learning for Networks,
    Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge
    Discovery and Data Mining, https://arxiv.org/abs/1607.00653

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    start_vertices: int or list or cudf.Series

    max_depth: int, optional
        The maximum depth of the random walks

    use_padding: bool, optional

    p: double, optional
        Return factor, which represents the likelihood of backtracking to
        a previous node in the walk. A higher value makes it less likely to
        sample a previously visited node, while a lower value makes it more
        likely to backtrack, making the walk "local"

    q: double, optional
        In-out factor, which represents the likelihood of visiting nodes
        closer or further from the outgoing node. If q > 1, the random walk
        is likelier to visit nodes closer to the outgoing node. If q < 1, the
        random walk is likelier to visit nodes further from the outgoing node.

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
    >>> _, _, _ = cugraph.node2vec(G, sources, 3, True, 0.8, 0.5)

    """

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    weights = G.edgelist.edgelist_df['weights']

    srcs = cupy.asarray(srcs)
    dsts = cupy.asarray(dsts)
    weights = cupy.asarray(weights)
    sources = cupy.asarray(sources)

    resource_handle = pylibcugraph.experimental.ResourceHandle()
    graph_props = pylibcugraph.experimental.GraphProperties(
                    is_multigraph=G.is_multigraph())

    # FIXME: remove later
    store_transposed = False
    renumber = False
    do_expensive_check = False

    SGGraph = pylibcugraph.experimental.SGGraph(resource_handle, graph_props,
                                                srcs, dsts, weights,
                                                store_transposed, renumber,
                                                do_expensive_check)

    vertex_set, edge_set, sizes = pylibcugraph.experimental.node2vec(
                                    resource_handle, SGGraph, sources,
                                    max_depth, use_padding, p, q)

    # Do prep work for start_vertices in case G is renumbered.

    # Call pylibcugraph wrapper

    # Undo renumbering and deal with padding
    return vertex_set, edge_set, sizes
