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

from cugraph.community import subgraph_extraction_wrapper
from cugraph.structure.graph import null_check, DiGraph


def subgraph(G, vertices):
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works for both directed and undirected graphs,
    it does not actually traverse the edges, simply pulls out any edges that
    are incident on vertices that are both contained in the vertices list.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor
    vertices : cudf.Series
        Specifies the vertices of the induced subgraph

    Returns
    -------
    Sg : cugraph.Graph
        A graph object containing the subgraph induced by the given vertex set.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> verts = numpy.zeros(3, dtype=numpy.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg = cugraph.subgraph(G, sverts)
    """

    null_check(vertices)

    result_graph = DiGraph()

    subgraph_extraction_wrapper.subgraph(
        G,
        vertices,
        result_graph)

    return result_graph
