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
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )
import cudf


def katz_centrality_2(
    G, alpha=None, beta=1.0, max_iter=1000, tol=1.0e-6,
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

    alpha : float, optional (default=None)
        Attenuation factor defaulted to None. If alpha is not specified then
        it is internally calculated as 1/(degree_max) where degree_max is the
        maximum out degree.

        NOTE:
            The maximum acceptable value of alpha for convergence
            alpha_max = 1/(lambda_max) where lambda_max is the largest
            eigenvalue of the graph.
            Since lambda_max is always lesser than or equal to degree_max for a
            graph, alpha_max will always be greater than or equal to
            (1/degree_max). Therefore, setting alpha to (1/degree_max) will
            guarantee that it will never exceed alpha_max thus in turn
            fulfilling the requirement for convergence.

    beta : float, optional (default=1.0)
        Weight scalar added to each vertex's new Katz Centrality score in every
        iteration

    max_iter : int, optional (default=1000)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 1000.

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

    # breakpoint()
    srcs = G.edgelist.edgelist_df['src']
    dsts = G.edgelist.edgelist_df['dst']
    if 'weights' in G.edgelist.edgelist_df.columns:
        weights = G.edgelist.edgelist_df['weights']
    else:
        weights = cudf.Series((srcs + 1) / (srcs + 1), dtype=srcs.dtype)

    if alpha is None or alpha <= 0.0:
        largest_out_degree = G.degrees().nlargest(n=1, columns="out_degree")
        largest_out_degree = largest_out_degree["out_degree"].iloc[0]
        alpha = 1 / (largest_out_degree + 1)

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

    vertices, values = katz_centrality(resource_handle, sg, nstart, alpha,
                                       beta, tol, max_iter,
                                       do_expensive_check)

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["katz_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "katz_centrality")
        return dict
    else:
        return df
