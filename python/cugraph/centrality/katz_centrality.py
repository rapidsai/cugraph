# Copyright (c) 2019, NVIDIA CORPORATION.
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


def katz_centrality(G,
                    alpha=0.1,
                    max_iter=100,
                    tol=1.0e-6,
                    nstart=None,
                    normalized=True):
    """
    Compute the Katz centrality for the nodes of the graph G. cuGraph does not
    currently support the 'beta' and 'weight' parameters as seen in the
    corresponding networkX call. This implementation is based on a relaxed
    version of Katz defined by Foster with a reduced computational complexity
    of O(n+m)

    Foster, K.C., Muth, S.Q., Potterat, J.J. et al.
    Computational & Mathematical Organization Theory (2001) 7: 275.
    https://doi.org/10.1023/A:1013470632383

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges where undirected edges are
        represented as directed edges in both directions.
    alpha : float
        Attenuation factor with a default value of 0.1.  If alpha is not less
        than 1/(lambda_max) where lambda_max is the maximum degree
        GDF_CUDA_ERROR is returned
    max_iter : int
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.
    tolerance : float
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.
    nstart : cudf.Dataframe
        GPU Dataframe containing the initial guess for katz centrality.

        nstart['vertex'] : cudf.Series
            Contains the vertex identifiers
        nstart['values'] : cudf.Series
            Contains the katz centrality values of vertices

    normalized : bool
        If True normalize the resulting katz centrality values

    Returns
    -------
    df : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding katz centrality values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['katz_centrality'] : cudf.Series
            Contains the katz centrality of vertices

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> kc = cugraph.katz_centrality(G)
    """

    df = katz_centrality_wrapper.katz_centrality(G, alpha, max_iter,
                                                 tol, nstart, normalized)

    return df
