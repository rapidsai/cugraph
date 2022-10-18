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

from pylibcugraph import (
    eigenvector_centrality as pylib_eigen,
    ResourceHandle,
)
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
import cudf
import warnings


def eigenvector_centrality(G, max_iter=100, tol=1.0e-6):
    """
    Compute the eigenvector centrality for a graph G.

    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node i is the
    i-th element of the vector x defined by the eigenvector equation.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.

    tol : float, optional (default=1e-6)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.

    normalized : not supported
        If True normalize the resulting eigenvector centrality values

    Returns
    -------
    df : cudf.DataFrame or Dictionary if using NetworkX
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding eigenvector centrality values.
        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['eigenvector_centrality'] : cudf.Series
            Contains the eigenvector centrality of vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> ec = cugraph.eigenvector_centrality(G)

    """
    if (not isinstance(max_iter, int)) or max_iter <= 0:
        raise ValueError(f"'max_iter' must be a positive integer" f", got: {max_iter}")
    if (not isinstance(tol, float)) or (tol <= 0.0):
        raise ValueError(f"'tol' must be a positive float, got: {tol}")

    G, isNx = ensure_cugraph_obj_for_nx(G, store_transposed=True)
    if G.store_transposed is False:
        warning_msg = (
            "Eigenvector centrality expects the 'store_transposed' "
            "flag to be set to 'True' for optimal performance "
            "during the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    vertices, values = pylib_eigen(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        epsilon=tol,
        max_iterations=max_iter,
        do_expensive_check=False,
    )

    vertices = cudf.Series(vertices)
    values = cudf.Series(values)

    df = cudf.DataFrame()
    df["vertex"] = vertices
    df["eigenvector_centrality"] = values

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        dict = df_score_to_dictionary(df, "eigenvector_centrality")
        return dict
    else:
        return df
