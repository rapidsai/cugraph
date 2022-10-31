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

import cudf


# FIXME: PLC uniform random walks returns a padded results.
# Should we also support the unpadded option?
def uniform_random_walks(G, start_vertices, max_depth=None):
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

    Returns
    -------
    vertex_paths : cudf.Series or cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    max_path_length : int
        The maximum path length

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> M = karate.get_edgelist(fetch=True)
    >>> G = karate.get_graph()
    >>> _, _, _ = cugraph.uniform_random_walks(G, M, 3)

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

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices)

    if G.renumbered is True:
        if isinstance(start_vertices, cudf.DataFrame):
            start_vertices = G.lookup_internal_vertex_id(
                start_vertices, start_vertices.columns
            )
        else:
            start_vertices = G.lookup_internal_vertex_id(start_vertices)

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

    edge_set = cudf.Series(edge_set)

    return vertex_set, edge_set, max_path_length
