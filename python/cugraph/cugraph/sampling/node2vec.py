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

import pylibcugraph, cupy
# import numpy, cudf

def node2vec(G, sources, max_depth, use_padding, p=1.0, q=1.0):
    """
    Computes node2vec.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    sources: cudf.Series

    max_depth: int, optional

    use_padding: bool, optional

    p: double, optional

    q: double, optional

    Returns
    -------

    Example
    -------
    >>> M = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
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

    SGGraph = pylibcugraph.experimental.SGGraph(resource_handle, graph_props,
                                                srcs, dsts, weights,
                                                store_transposed, renumber,
                                                do_expensive_check)
    
    vertex_set, edge_set, sizes = pylibcugraph.experimental.node2vec(
                                    resource_handle, SGGraph, sources, max_depth,
                                    use_padding, p, q)

    # Do prep work for start_vertices in case G is renumbered.

    # Call pylibcugraph wrapper

    # Undo renumbering and deal with padding
    return vertex_set, edge_set, sizes
