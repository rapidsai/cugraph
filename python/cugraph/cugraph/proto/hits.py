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

from cugraph.link_analysis import hits_wrapper
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               df_score_to_dictionary,
                               )


def hits(G, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    The cuGraph implementation of HITS is a wrapper around the gunrock
    implementation of HITS.

    Note that the gunrock implementation uses a 2-norm, while networkx
    uses a 1-norm.  The raw scores will be different, but the rank ordering
    should be comparable with networkx.

    Parameters
    ----------
    graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        The adjacency list will be computed if not already present.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.
        The gunrock implementation does not currently support tolerance,
        so this will in fact be the number of iterations the HITS algorithm
        executes.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.  This parameter is not currently supported.

    nstart : cudf.Dataframe, optional (default=None)
        Not currently supported

    normalized : bool, optional (default=True)
        Not currently supported, always used as True

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
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> hits = cugraph.hits(G, max_iter = 50)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G)

    df = hits_wrapper.hits(G, max_iter, tol)

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        d1 = df_score_to_dictionary(df[["vertex", "hubs"]], "hubs")
        d2 = df_score_to_dictionary(df[["vertex", "authorities"]],
                                    "authorities")
        df = (d1, d2)

    return df
