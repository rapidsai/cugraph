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

import cudf
from cugraph.structure import Graph
from cugraph.community import subgraph_extraction_wrapper
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    cugraph_to_nx,
)


def subgraph(G, vertices):
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works with both directed and undirected graphs
    and does not actually traverse the edges, but instead simply pulls out any
    edges that are incident on vertices that are both contained in the vertices
    list.

    Parameters
    ----------
    G : cugraph.Graph
        cuGraph graph descriptor
        The current implementation only supports weighted graphs.

    vertices : cudf.Series or cudf.DataFrame
        Specifies the vertices of the induced subgraph. For multi-column
        vertices, vertices should be provided as a cudf.DataFrame

    Returns
    -------
    Sg : cugraph.Graph
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

    G, isNx = ensure_cugraph_obj_for_nx(G)
    directed = G.is_directed()

    if G.renumbered:
        if isinstance(vertices, cudf.DataFrame):
            vertices = G.lookup_internal_vertex_id(vertices, vertices.columns)
        else:
            vertices = G.lookup_internal_vertex_id(vertices)

    result_graph = Graph(directed=directed)

    df = subgraph_extraction_wrapper.subgraph(G, vertices)
    src_names = "src"
    dst_names = "dst"

    if G.renumbered:
        df, src_names = G.unrenumber(df, src_names, get_column_names=True)
        df, dst_names = G.unrenumber(df, dst_names, get_column_names=True)

    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names, destination=dst_names)

    if isNx is True:
        result_graph = cugraph_to_nx(result_graph)

    return result_graph
