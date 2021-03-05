# Copyright (c) 2021, NVIDIA CORPORATION.
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

from cugraph.traversal import traveling_salesperson_wrapper
from cugraph.structure.graph import null_check
import cudf


def traveling_salesperson(pos_list,
                          restarts=100000,
                          beam_search=True,
                          k=4,
                          nstart=None,
                          verbose=False,
                          ):
    """
    Finds an approximate solution to the traveling salesperson problem (TSP).
    cuGraph computes an approximation of the TSP problem using hill climbing
    optimization.

    The current implementation does not support a weighted graph.
    Parameters
    ----------
    pos_list: cudf.DataFrame
        Data frame with initial vertex positions containing three columns:
        'vertex' ids and 'x', 'y' positions.
    restarts: int
        Number of starts to try. The more restarts, the better the solution
        will be approximated. The number of restarts depends on the problem
        size and should be kept low for instances above 2k cities.
    beam_search: bool
        Specify if the initial solution should use KNN for an approximation
        solution.
    k: int
        Beam width to use in the search.
    nstart: int
        Vertex id to use as starting position.
    verbose: bool
        Logs configuration and iterative improvement.

    Returns
    -------
    route : cudf.Series
        cudf.Series of size V containing the ordered list of vertices
        than needs to be visited.
    """

    if not isinstance(pos_list, cudf.DataFrame):
        raise TypeError("Instance should be cudf.DataFrame")

    null_check(pos_list['vertex'])
    null_check(pos_list['x'])
    null_check(pos_list['y'])

    if nstart is not None and not pos_list[pos_list['vertex'] == nstart].index:
        raise ValueError("nstart should be in vertex ids")

    route, cost = traveling_salesperson_wrapper.traveling_salesperson(
            pos_list,
            restarts,
            beam_search,
            k,
            nstart,
            verbose)
    return route, cost
