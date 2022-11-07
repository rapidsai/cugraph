# Copyright (c) 2022, NVIDIA CORPORATION.
#
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
#

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
from pylibcugraph import ResourceHandle, hits as pylibcugraph_hits
import cudf
import warnings


def hits(G, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
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
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> hits = cugraph.hits(G, max_iter = 50)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G, store_transposed=True)
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

    if isNx is True:
        d1 = df_score_to_dictionary(results[["vertex", "hubs"]], "hubs")
        d2 = df_score_to_dictionary(results[["vertex", "authorities"]], "authorities")
        results = (d1, d2)

    if G.renumbered:
        results = G.unrenumber(results, "vertex")

    return results
