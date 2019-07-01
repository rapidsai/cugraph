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

import cugraph
import cugraph.community.subgraph_extraction_wrapper as cpp_subgraph_extraction


def subgraph(G, vertices):
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works for both directed and undirected graphs,
    it does not actually traverse the edges, simply pulls out any edges that
    are incident on vertices that are both contained in the vertices list.

    Parameters
    ----------
    G : cuGraph.Graph
       cuGraph graph descriptor
    vertices : cudf.Series
        Specifies the vertices of the induced subgraph

    Returns
    -------
    Sg : cuGraph.Graph
        A graph object containing the subgraph induced by the given vertex set.

    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> verts = numpy.zeros(3, dtype=np.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg = cuGraph.subgraph(G, sverts)
    """

    cugraph.null_check(vertices)

    result_graph = cugraph.Graph()

    cpp_subgraph_extraction.subgraph(
        G.graph_ptr,
        vertices,
        result_graph.graph_ptr)

    return result_graph
