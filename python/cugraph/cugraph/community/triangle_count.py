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

from cugraph.community import triangle_count_wrapper
from cugraph.utilities import ensure_cugraph_obj_for_nx
import warnings


def triangles(G):
    """
    Compute the number of triangles (cycles of length three) in the
    input graph.

    Unlike NetworkX, this algorithm simply returns the total number of
    triangle and not the number per vertex.

    Parameters
    ----------
    G : cugraph.graph or networkx.Graph
        cuGraph graph descriptor, should contain the connectivity information,
        (edge weights are not used in this algorithm).
        The current implementation only supports undirected graphs.

    Returns
    -------
    count : int64
        A 64 bit integer whose value gives the number of triangles in the
        graph.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> count = cugraph.triangles(G)

    """
    warning_msg = ("This call is deprecated and will be refactored "
                   "in the next release")
    warnings.warn(warning_msg, PendingDeprecationWarning)

    G, _ = ensure_cugraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    result = triangle_count_wrapper.triangles(G)

    return result
