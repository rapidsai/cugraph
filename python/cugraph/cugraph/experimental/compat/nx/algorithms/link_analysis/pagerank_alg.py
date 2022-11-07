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
import cugraph.utilities
import cudf
import numpy as np


def create_cudf_from_dict(dict_in):
    """
    converts python dictionary to a cudf.Dataframe as needed by this
    cugraph pagerank call.

    Parameters
    ----------
    dictionary with node ids(key) and values

    Returns
    -------
    a cudf DataFrame of (vertex)ids and values.
    """
    if not (isinstance(dict_in, dict)):
        raise TypeError("type_name must be a dict, got: " f"{type(dict_in)}")
    # FIXME: Looking to replacing fromiter with rename and
    # compare performance
    k = np.fromiter(dict_in.keys(), dtype="int32")
    v = np.fromiter(dict_in.values(), dtype="float32")
    df = cudf.DataFrame({"vertex": k, "values": v})
    return df


def pagerank(
    G,
    alpha=0.85,
    personalization=None,
    max_iter=100,
    tol=1.0e-6,
    nstart=None,
    weight="weight",
    dangling=None,
):

    """
    Calls the cugraph pagerank algorithm taking in a networkX object.
    In future releases it will maintain compatibility but will migrate more
    of the workflow to the GPU.

    Parameters
    ----------
    G : networkx.Graph

    alpha : float, optional (default=0.85)
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.

    personalization : dictionary, optional (default=None)
        dictionary comes from networkx is converted to a dataframe
        containing the personalization information.

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

    nstart : dictionary, optional (default=None)
        dictionary containing the initial guess vertex and value for pagerank.
        Will be converted to a Dataframe before calling the cugraph algorithm
        nstart['vertex'] : cudf.Series
            Subset of vertices of graph for initial guess for pagerank values
        nstart['values'] : cudf.Series
            Pagerank values for vertices

    weight: str, optional (default=None)
        This parameter is here for NetworkX compatibility and not
        yet supported in this algorithm

    dangling : dict, optional (default=None)
        This parameter is here for NetworkX compatibility and ignored

    Returns
    -------
        PageRank : dictionary
               A dictionary of nodes with the PageRank as value

    """
    local_pers = None
    local_nstart = None
    if personalization is not None:
        local_pers = create_cudf_from_dict(personalization)
    if nstart is not None:
        local_nstart = create_cudf_from_dict(nstart)
    return cugraph.pagerank(
        G,
        alpha=alpha,
        personalization=local_pers,
        max_iter=max_iter,
        tol=tol,
        nstart=local_nstart,
        weight=weight,
        dangling=dangling,
    )
