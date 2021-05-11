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

# FIXME might be more efficient to return either (df + offset) or 3 cudf.Series


def random_walks(
    G,
    start_vertices,
    max_depth=None,
    use_padding = False):
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


    Returns
    -------
    random_walks_edge_lists : cudf.DataFrame
        GPU data frame containing all random walks sources identifiers,
        destination identifiers, edge weights

    seeds_offsets: cudf.Series
        Series containing the starting offset in the returned edge list
        for each vertex in start_vertices.
    """
    if max_depth is None:
        raise TypeError("must specify a 'max_depth'")

    G, _ = cugraph.utilities.check_nx_graph(G)

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

    return vertex_set, edge_set, sizes


def rw_path(num_paths, sizes):
    return random_walks_wrapper.rw_path_retrieval(num_paths, sizes)
