# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from cugraph.structure import Graph
from pylibcugraph import ResourceHandle, hits as pylibcugraph_hits
import cudf
import warnings


def hits(
    G: Graph, max_iter=100, tol=1.0e-5, nstart=None, normalized=True
) -> cudf.DataFrame:
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    Both cuGraph and networkx implementation use a 1-norm.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        The adjacency list will be computed if not already present.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.

    nstart : cudf.Dataframe, optional (default=None)
        The initial hubs guess vertices along with their initial hubs guess
        value

        nstart['vertex'] : cudf.Series
            Initial hubs guess vertices
        nstart['values'] : cudf.Series
            Initial hubs guess values

    normalized : bool, optional (default=True)
        A flag to normalize the results

    Returns
    -------
    HubsAndAuthorities : cudf.DataFrame
        GPU data frame containing three cudf.Series of size V: the vertex
        identifiers and the corresponding hubs values and the corresponding
        authorities values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['hubs'] : cudf.Series
            Contains the hubs score
        df['authorities'] : cudf.Series
            Contains the authorities score


    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> hits = cugraph.hits(G, max_iter = 50)

    """

    if G.store_transposed is False:
        warning_msg = (
            "HITS expects the 'store_transposed' flag "
            "to be set to 'True' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    do_expensive_check = False
    init_hubs_guess_vertices = None
    init_hubs_guess_values = None

    if nstart is not None:
        init_hubs_guess_vertices = nstart["vertex"]
        init_hubs_guess_values = nstart["values"]

    vertices, hubs, authorities = pylibcugraph_hits(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        tol=tol,
        max_iter=max_iter,
        initial_hubs_guess_vertices=init_hubs_guess_vertices,
        initial_hubs_guess_values=init_hubs_guess_values,
        normalized=normalized,
        do_expensive_check=do_expensive_check,
    )
    results = cudf.DataFrame()
    results["vertex"] = cudf.Series(vertices)
    results["hubs"] = cudf.Series(hubs)
    results["authorities"] = cudf.Series(authorities)

    if G.renumbered:
        results = G.unrenumber(results, "vertex")

    return results
