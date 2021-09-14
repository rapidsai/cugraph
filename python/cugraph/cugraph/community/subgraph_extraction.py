# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from cugraph.utilities import check_nx_graph
import cudf
from cugraph.utilities import cugraph_to_nx


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
    vertices : cudf.Series or cudf.DataFrame
        Specifies the vertices of the induced subgraph. For multi-column
        vertices, vertices should be provided as a cudf.DataFrame

    Returns
    -------
    Sg : cugraph.Graph
        A graph object containing the subgraph induced by the given vertex set.

    Examples
    --------
    >>> gdf = cudf.read_csv('datasets/karate.csv',
                          delimiter = ' ',
                          dtype=['int32', 'int32', 'float32'],
                          header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> verts = numpy.zeros(3, dtype=numpy.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg = cugraph.subgraph(G, sverts)
    """

    G, isNx = check_nx_graph(G)

    if G.renumbered:
        if isinstance(vertices, cudf.DataFrame):
            vertices = G.lookup_internal_vertex_id(vertices, vertices.columns)
        else:
            vertices = G.lookup_internal_vertex_id(vertices)

    result_graph = type(G)()

    df = subgraph_extraction_wrapper.subgraph(G, vertices)

    if G.renumbered:
        df, src_names = G.unrenumber(df, "src", get_column_names=True)
        df, dst_names = G.unrenumber(df, "dst", get_column_names=True)

    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names,
            edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names,
                                        destination=dst_names)

    if isNx is True:
        result_graph = cugraph_to_nx(result_graph)

    return result_graph
