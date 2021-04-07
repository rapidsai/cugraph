# Copyright (c) 2021, NVIDIA CORPORATION.
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
import cugraph
from collections import defaultdict


def random_walks(
    G,
    start_vertices,
    max_depth=None
):
    """
    compute random walks for each nodes in 'start_vertices'

    parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph can be either directed (DiGraph) or undirected (Graph).
        Weights in the graph are ignored.
        Use weight parameter if weights need to be considered
        (currently not supported)

    start_vertices : int or list or cudf.Series
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks

    max_depth : int
        The depth of the random walks


    Returns
    -------
    random_walks_edge_lists : cudf.DataFrame
        GPU data frame containing all random walks sources identifiers,
        destination identifiers, edge weights

    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list
        for each seed.
    """
    if max_depth is None:
        raise TypeError("must specify a 'max_depth'")

    G, _ = cugraph.utilities.check_nx_graph(G)

    if start_vertices is int:
        start_vertices = [start_vertices]

    if not isinstance(start_vertices, cudf.Series):
        start_vertices = cudf.Series(start_vertices)

    if G.renumbered is True:
        start_vertices = G.lookup_internal_vertex_id(start_vertices)
    vertex_set, edge_set, sizes = random_walks_wrapper.random_walks(
        G, start_vertices, max_depth)

    if G.renumbered:
        df_ = cudf.DataFrame()
        df_['vertex_set'] = vertex_set
        df_ = G.unrenumber(df_, 'vertex_set', preserve_order=True)
        vertex_set = cudf.Series(df_['vertex_set'])

    edge_list = defaultdict(list)
    next_path_idx = 0
    offsets = [0]

    df = cudf.DataFrame()
    for s in sizes.values_host:
        for i in range(next_path_idx, s+next_path_idx-1):
            edge_list['src'].append(vertex_set.values_host[i])
            edge_list['dst'].append(vertex_set.values_host[i+1])
        next_path_idx += s
        df = df.append(edge_list, ignore_index=True)
        offsets.append(df.index[-1]+1)
        edge_list['src'].clear()
        edge_list['dst'].clear()
    df['weight'] = edge_set
    offsets = cudf.Series(offsets)

    return df, offsets
