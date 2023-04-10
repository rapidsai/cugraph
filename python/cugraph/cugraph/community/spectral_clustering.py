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

from cugraph.community import spectral_clustering_wrapper
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)


def spectralBalancedCutClustering(
    G,
    num_clusters,
    num_eigen_vects=2,
    evs_tolerance=0.00001,
    evs_max_iter=100,
    kmean_tolerance=0.00001,
    kmean_max_iter=100,
):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    balanced cut method.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        Graph descriptor

    num_clusters : integer
        Specifies the number of clusters to find, must be greater than 1

    num_eigen_vects : integer, optional
        Specifies the number of eigenvectors to use. Must be lower or equal to
        num_clusters. Default is 2

    evs_tolerance: float, optional
        Specifies the tolerance to use in the eigensolver.
        Default is 0.00001

    evs_max_iter: integer, optional
        Specifies the maximum number of iterations for the eigensolver.
        Default is 100

    kmean_tolerance: float, optional
        Specifies the tolerance to use in the k-means solver.
        Default is 0.00001

    kmean_max_iter: integer, optional
        Specifies the maximum number of iterations for the k-means solver.
        Default is 100

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding cluster assignments.

        df['vertex'] : cudf.Series
            contains the vertex identifiers
        df['cluster'] : cudf.Series
            contains the cluster assignments

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.spectralBalancedCutClustering(G, 5)

    """

    # Error checking in C++ code

    G, isNx = ensure_cugraph_obj_for_nx(G)

    df = spectral_clustering_wrapper.spectralBalancedCutClustering(
        G,
        num_clusters,
        num_eigen_vects,
        evs_tolerance,
        evs_max_iter,
        kmean_tolerance,
        kmean_max_iter,
    )

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "cluster")

    return df


def spectralModularityMaximizationClustering(
    G,
    num_clusters,
    num_eigen_vects=2,
    evs_tolerance=0.00001,
    evs_max_iter=100,
    kmean_tolerance=0.00001,
    kmean_max_iter=100,
):
    """
    Compute a clustering/partitioning of the given graph using the spectral
    modularity maximization method.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        cuGraph graph descriptor. This graph should have edge weights.

    num_clusters : integer
        Specifies the number of clusters to find

    num_eigen_vects : integer, optional
        Specifies the number of eigenvectors to use. Must be lower or equal to
        num_clusters.  Default is 2

    evs_tolerance: float, optional
        Specifies the tolerance to use in the eigensolver.
        Default is 0.00001

    evs_max_iter: integer, optional
        Specifies the maximum number of iterations for the eigensolver.
        Default is 100

    kmean_tolerance: float, optional
        Specifies the tolerance to use in the k-means solver.
        Default is 0.00001

    kmean_max_iter: integer, optional
        Specifies the maximum number of iterations for the k-means solver.
        Default is 100

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding cluster assignments.

        df['vertex'] : cudf.Series
            contains the vertex identifiers
        df['cluster'] : cudf.Series
            contains the cluster assignments

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.spectralModularityMaximizationClustering(G, 5)

    """

    # Error checking in C++ code

    G, isNx = ensure_cugraph_obj_for_nx(G)

    df = spectral_clustering_wrapper.spectralModularityMaximizationClustering(
        G,
        num_clusters,
        num_eigen_vects,
        evs_tolerance,
        evs_max_iter,
        kmean_tolerance,
        kmean_max_iter,
    )

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "cluster")

    return df


def analyzeClustering_modularity(
    G, n_clusters, clustering, vertex_col_name="vertex", cluster_col_name="cluster"
):
    """
    Compute the modularity score for a given partitioning/clustering.
    The assumption is that “clustering” is the results from a call
    from a special clustering algorithm and contains columns named
    “vertex” and “cluster”.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        graph descriptor. This graph should have edge weights.

    n_clusters : integer
        Specifies the number of clusters in the given clustering

    clustering : cudf.DataFrame
        The cluster assignment to analyze.

    vertex_col_name : str or list of str, optional (default='vertex')
        The names of the column in the clustering dataframe identifying
        the external vertex id

    cluster_col_name : str, optional (default='cluster')
        The name of the column in the clustering dataframe identifying
        the cluster id

    Returns
    -------
    score : float
        The computed modularity score

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.spectralBalancedCutClustering(G, 5)
    >>> score = cugraph.analyzeClustering_modularity(G, 5, df)

    """
    if type(vertex_col_name) is list:
        if not all(isinstance(name, str) for name in vertex_col_name):
            raise Exception("vertex_col_name must be list of string")
    elif type(vertex_col_name) is not str:
        raise Exception("vertex_col_name must be a string")

    if type(cluster_col_name) is not str:
        raise Exception("cluster_col_name must be a string")

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if G.renumbered:
        clustering = G.add_internal_vertex_id(
            clustering, "vertex", vertex_col_name, drop=True
        )

    clustering = clustering.sort_values("vertex")

    score = spectral_clustering_wrapper.analyzeClustering_modularity(
        G, n_clusters, clustering[cluster_col_name]
    )

    return score


def analyzeClustering_edge_cut(
    G, n_clusters, clustering, vertex_col_name="vertex", cluster_col_name="cluster"
):
    """
    Compute the edge cut score for a partitioning/clustering
    The assumption is that “clustering” is the results from a call
    from a special clustering algorithm and contains columns named
    “vertex” and “cluster”.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor

    n_clusters : integer
        Specifies the number of clusters in the given clustering

    clustering : cudf.DataFrame
        The cluster assignment to analyze.

    vertex_col_name : str, optional (default='vertex')
        The name of the column in the clustering dataframe identifying
        the external vertex id

    cluster_col_name : str, optional (default='cluster')
        The name of the column in the clustering dataframe identifying
        the cluster id

    Returns
    -------
    score : float
        The computed edge cut score

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.spectralBalancedCutClustering(G, 5)
    >>> score = cugraph.analyzeClustering_edge_cut(G, 5, df)

    """
    if type(vertex_col_name) is list:
        if not all(isinstance(name, str) for name in vertex_col_name):
            raise Exception("vertex_col_name must be list of string")
    elif type(vertex_col_name) is not str:
        raise Exception("vertex_col_name must be a string")

    if type(cluster_col_name) is not str:
        raise Exception("cluster_col_name must be a string")

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if G.renumbered:
        clustering = G.add_internal_vertex_id(
            clustering, "vertex", vertex_col_name, drop=True
        )

    clustering = clustering.sort_values("vertex").reset_index(drop=True)

    score = spectral_clustering_wrapper.analyzeClustering_edge_cut(
        G, n_clusters, clustering[cluster_col_name]
    )

    return score


def analyzeClustering_ratio_cut(
    G, n_clusters, clustering, vertex_col_name="vertex", cluster_col_name="cluster"
):
    """
    Compute the ratio cut score for a partitioning/clustering

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor. This graph should have edge weights.

    n_clusters : integer
        Specifies the number of clusters in the given clustering

    clustering : cudf.DataFrame
        The cluster assignment to analyze.

    vertex_col_name : str, optional (default='vertex')
        The name of the column in the clustering dataframe identifying
        the external vertex id

    cluster_col_name : str, optional (default='cluster')
        The name of the column in the clustering dataframe identifying
        the cluster id

    Returns
    -------
    score : float
        The computed ratio cut score

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.spectralBalancedCutClustering(G, 5)
    >>> score = cugraph.analyzeClustering_ratio_cut(G, 5, df, 'vertex',
    ...                                             'cluster')

    """
    if type(vertex_col_name) is list:
        if not all(isinstance(name, str) for name in vertex_col_name):
            raise Exception("vertex_col_name must be list of string")
    elif type(vertex_col_name) is not str:
        raise Exception("vertex_col_name must be a string")

    if type(cluster_col_name) is not str:
        raise Exception("cluster_col_name must be a string")

    if G.renumbered:
        clustering = G.add_internal_vertex_id(
            clustering, "vertex", vertex_col_name, drop=True
        )

    clustering = clustering.sort_values("vertex")

    score = spectral_clustering_wrapper.analyzeClustering_ratio_cut(
        G, n_clusters, clustering[cluster_col_name]
    )

    return score
