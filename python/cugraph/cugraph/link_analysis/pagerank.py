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

import cudf

from cugraph.link_analysis import pagerank_wrapper
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )


def pagerank(
    G, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-5, nstart=None,
    weight=None, dangling=None
):
    """
    Find the PageRank score for every vertex in a graph. cuGraph computes an
    approximation of the Pagerank eigenvector using the power method. The
    number of iterations depends on the properties of the network itself; it
    increases when the tolerance descreases and/or alpha increases toward the
    limiting value of 1. The user is free to use default values or to provide
    inputs for the initial guess, tolerance and maximum number of iterations.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list.
        The transposed adjacency list will be computed if not already present.

    alpha : float, optional (default=0.85)
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    personalization : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the personalization information.

        personalization['vertex'] : cudf.Series
            Subset of vertices of graph for personalization
        personalization['values'] : cudf.Series
            Personalization values for vertices

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.

    tol : float, optional (default=1e-05)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0E-5.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 0.01 and 0.00001 are
        acceptable.

    nstart : cudf.Dataframe, optional (default=None)
        GPU Dataframe containing the initial guess for pagerank.

        nstart['vertex'] : cudf.Series
            Subset of vertices of graph for initial guess for pagerank values
        nstart['values'] : cudf.Series
            Pagerank values for vertices

    weight: str, optional (default=None)
        The attribute column to be used as edge weights if Graph is a NetworkX
        Graph. This parameter is here for NetworkX compatibility and is ignored
        in case of a cugraph.Graph

    dangling : dict, optional (default=None)
        This parameter is here for NetworkX compatibility and ignored

    Returns
    -------
    PageRank : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding PageRank values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['pagerank'] : cudf.Series
            Contains the PageRank score


    Examples
    --------
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500, tol = 1.0e-05)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G, weight)

    if personalization is not None:
        if not isinstance(personalization, cudf.DataFrame):
            raise NotImplementedError(
                "personalization other than a cudf dataframe "
                "currently not supported"
            )
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = personalization.columns[:-1].to_list()
            else:
                cols = 'vertex'
            personalization = G.add_internal_vertex_id(
                personalization, "vertex", cols
            )

    if nstart is not None:
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = nstart.columns[:-1].to_list()
            else:
                cols = 'vertex'
            nstart = G.add_internal_vertex_id(
                nstart, "vertex", cols
            )

    df = pagerank_wrapper.pagerank(
        G, alpha, personalization, max_iter, tol, nstart
    )

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        return df_score_to_dictionary(df, 'pagerank')
    else:
        return df
