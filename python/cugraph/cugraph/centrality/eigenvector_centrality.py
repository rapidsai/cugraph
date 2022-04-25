# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
                                       eigenvector_centrality as pylibcugraph_eigenvector
                                       )
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )
import cudf


def eigenvector_centrality(
    G, max_iter=1000, tol=1.0e-6, nstart=None, normalized=True
):
    """
    Compute the eigenvector centrality for a graph G.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    max_iter :

    tol : float

    nstart :

    normalized :

    """
    # Checks

    G, isNx = ensure_cugraph_obj_for_nx(G)

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    if 'weights' in G.edgelist.edgelist_df.columns:
        weights = G.edgelist.edgelist_df['weights']
    else:
        # FIXME: If weights column is not imported, a weights column of 1s
        # with type hardcoded to float32 is passed into wrapper
        weights = cudf.Series((srcs + 1) / (srcs + 1), dtype="float32")

    if nstart is not None:
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = nstart.columns[:-1].to_list()
            else:
                cols = 'vertex'
            nstart = G.add_internal_vertex_id(nstart, 'vertex', cols)
            nstart = nstart[nstart.columns[0]]

    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_multigraph=G.is_multigraph())
    store_transposed = False
    renumber = False
    do_expensive_check = False

    sg = SGGraph(resource_handle, graph_props, srcs, dsts, weights,
                 store_transposed, renumber, do_expensive_check)

    vertices, values = pylibcugraph_eigenvector(resource_handle, sg, nstart,
                                                tol, max_iter,
                                                do_expensive_check)

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["eigenvector_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "eigenvector_centrality")
        return dict
    else:
        return df
