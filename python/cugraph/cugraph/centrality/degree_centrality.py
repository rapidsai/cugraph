# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf


def degree_centrality(G, normalized=True) -> cudf.DataFrame:
    """
    Computes the degree centrality of each vertex of the input graph.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

    normalized : bool, optional, default=True
        If True normalize the resulting degree centrality values

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding degree centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers

        df['degree_centrality'] : cudf.Series
            Contains the degree centrality of vertices

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> dc = cugraph.degree_centrality(G)

    """
    df = G.degree()
    df.rename(columns={"degree": "degree_centrality"}, inplace=True)

    if normalized:
        df["degree_centrality"] /= G.number_of_nodes() - 1

    return df
