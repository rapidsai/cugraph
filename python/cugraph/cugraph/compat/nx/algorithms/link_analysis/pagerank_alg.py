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

import cugraph


def pagerank(
        G,
        alpha=0.85,
        personalization=None,
        max_iter=100,
        tol=1.0e-6,
        nstart=None,
        weight="weight",
        dangling=None):

    """
    Calls the cugraph pagerank algorithm taking in a networkX object.
    In future releases it will maintain compatibility but will migrate more
    of the workflow to the GPU.

    Parameters
    ----------
    G : networkx.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list.
        The transposed adjacency list will be computed if not already present.

    alpha : float, optional (default=0.85)
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    personalization : dictionary, optional (default=None)
        GPU Dataframe containing the personalization information.

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
        The attribute column to be used as edge weights.
        This parameter is here for NetworkX compatibility and is ignored
        in case of a cugraph.Graph

    dangling : dict, optional (default=None)
        This parameter is here for NetworkX compatibility and ignored

    Returns
    -------
        PageRank : dictionary
               A dictionary of nodes with the PageRank as value

    """
    print("Called compat.nx pagerank")
    return cugraph.pagerank(
            G,
            alpha,
            personalization,
            max_iter,
            tol,
            nstart,
            weight,
            dangling)
