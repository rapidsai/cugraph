# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
import warnings
import cudf
from cugraph.structure import Graph
from typing import Union
import networkx as nx


def subgraph(
    G,
    vertices: Union[cudf.Series, cudf.DataFrame],
) -> Union[Graph, nx.Graph]:
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works with both directed and undirected graphs
    and does not actually traverse the edges, but instead simply pulls out any
    edges that are incident on vertices that are both contained in the vertices
    list.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        The current implementation only supports weighted graphs.

    vertices : cudf.Series or cudf.DataFrame
        Specifies the vertices of the induced subgraph. For multi-column
        vertices, vertices should be provided as a cudf.DataFrame

    Returns
    -------
    Sg : cugraph.Graph or networkx.Graph
        A graph object containing the subgraph induced by the given vertex set.

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> verts = np.zeros(3, dtype=np.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg = cugraph.subgraph(G, sverts)

    """

    warning_msg = (
        "This call is deprecated and will be refactored "
        "in the next release. Please call 'cugraph.induced_subgraph()' instead"
    )
    warnings.warn(warning_msg, PendingDeprecationWarning)

    result_graph, _ = cugraph.induced_subgraph(G, vertices)

    return result_graph
