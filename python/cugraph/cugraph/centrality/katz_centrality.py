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

from cugraph.centrality import katz_centrality_wrapper
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )


def katz_centrality(
    G, alpha=None, beta=None, max_iter=100, tol=1.0e-6,
    nstart=None, normalized=True
):
    """
    Compute the Katz centrality for the nodes of the graph G. cuGraph does not
    currently support the 'beta' and 'weight' parameters as seen in the
    corresponding networkX call. This implementation is based on a relaxed
    version of Katz defined by Foster with a reduced computational complexity
    of O(n+m)

    On a directed graph, cuGraph computes the out-edge Katz centrality score.
    This is opposite of NetworkX which compute the in-edge Katz centrality
    score by default.  You can flip the NetworkX edges, using G.reverse,
    so that the results match cuGraph.

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
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed (DiGraph) or undirected edges (Graph).

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

    beta : float, optional (default=None)
        A weight scalar - currently Not Supported

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.

    tol : float, optional (default=1.0e-6)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    nstart : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the initial guess for katz centrality.

        nstart['vertex'] : cudf.Series
            Contains the vertex identifiers
        nstart['values'] : cudf.Series
            Contains the katz centrality values of vertices

    normalized : bool, optional, default=True
        If True normalize the resulting katz centrality values

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding katz centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['katz_centrality'] : cudf.Series
            Contains the katz centrality of vertices

    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> kc = cugraph.katz_centrality(G)

    """

    if beta is not None:
        raise NotImplementedError(
                "The beta argument is "
                "currently not supported"
        )

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if nstart is not None:
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = nstart.columns[:-1].to_list()
            else:
                cols = 'vertex'
            nstart = G.add_internal_vertex_id(nstart, 'vertex', cols)

    df = katz_centrality_wrapper.katz_centrality(
        G, alpha, max_iter, tol, nstart, normalized
    )

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, 'katz_centrality')
        return dict
    else:
        return df
