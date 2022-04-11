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
                                       katz_centrality
                                       )
from cugraph.utilities import ensure_cugraph_obj_for_nx
import cudf


def katz_centrality_2(
    G, alpha=0.1, beta=1.0, max_iter=1000, tol=1.0e-6,
    nstart=None, normalized=True
):
    """
    Compute the Katz centrality for the nodes of the graph G.

    References
    ----------
    Foster, K.C., Muth, S.Q., Potterat, J.J. et al.
    Computational & Mathematical Organization Theory (2001) 7: 275.
    https://doi.org/10.1023/A:1013470632383

    Katz, L. (1953). A new status index derived from sociometric analysis.
    Psychometrika, 18(1), 39-43.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph

    alpha : float, optional (default=0.1)

    beta : float, optional (default=1.0)

    max_iter : int, optional (default=1000)

    tol : float, optional (default=1e-6)

    nstart : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the initial guess for katz centrality.

        nstart['vertex'] : cudf.Series
            Contains the vertex identifiers
        nstart['values'] : cudf.Series
            Contains the katz centrality values of vertices

    normalized : bool, optional, default=True
        If True normalize the resulting katz centrality values

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> kc = cugraph.katz_centrality(G)

    """
    G, isNx = ensure_cugraph_obj_for_nx(G)

    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    weights = G.edgelist.edgelist_df['weights']
    # breakpoint()
    resource_handle = ResourceHandle()
    graph_props = GraphProperties(is_multigraph=G.is_multigraph())
    store_transposed = False
    renumber = False
    do_expensive_check = False

    sg = SGGraph(resource_handle, graph_props, srcs, dsts, weights,
                 store_transposed, renumber, do_expensive_check)

    vertices, values = katz_centrality(resource_handle, sg, nstart, alpha,
                                       beta, tol, max_iter,
                                       do_expensive_check)

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame(values, vertices)

    return df
