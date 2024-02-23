# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

import warnings
from typing import Union, Tuple

import cudf
from pylibcugraph import ResourceHandle
from pylibcugraph import induced_subgraph as pylibcugraph_induced_subgraph

from cugraph.structure import Graph
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    cugraph_to_nx,
)
from cugraph.utilities.utils import import_optional

# FIXME: the networkx.Graph type used in type annotations is specified
# using a string literal to avoid depending on and importing networkx.
# Instead, networkx is imported optionally, which may cause a problem
# for a type checker if run in an environment where networkx is not installed.
networkx = import_optional("networkx")


# FIXME: Move this function to the utility module so that it can be
# shared by other algos
def ensure_valid_dtype(input_graph: Graph, input: cudf.Series, input_name: str):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    input_dtype = input.dtype
    if input_dtype != vertex_dtype:
        warning_msg = (
            f"Subgraph requires '{input_name}' "
            "to match the graph's 'vertex' type. "
            f"input graph's vertex type is: {vertex_dtype} and got "
            f"'{input_name}' of type: "
            f"{input_dtype}."
        )
        warnings.warn(warning_msg, UserWarning)
        input = input.astype(vertex_dtype)

    return input


def induced_subgraph(
    G: Union[Graph, "networkx.Graph"],
    vertices: Union[cudf.Series, cudf.DataFrame],
    offsets: Union[list, cudf.Series] = None,
) -> Tuple[Union[Graph, "networkx.Graph"], cudf.Series]:
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works with both directed and undirected graphs
    and does not actually traverse the edges, but instead simply pulls out any
    edges that are incident on vertices that are both contained in the vertices
    list.

    If no subgraph can be extracted from the vertices provided, a 'None' value
    will be returned.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        The current implementation only supports weighted graphs.

    vertices : cudf.Series or cudf.DataFrame
        Specifies the vertices of the induced subgraph. For multi-column
        vertices, vertices should be provided as a cudf.DataFrame

    offsets : list or cudf.Series, optional
        Specifies the subgraph offsets into the subgraph vertices.
        If no offsets array is provided, a default array [0, len(vertices)]
        will be used.

    Returns
    -------
    Sg : cugraph.Graph or networkx.Graph
        A graph object containing the subgraph induced by the given vertex set.
    seeds_offsets: cudf.Series
        A cudf Series containing the starting offset in the returned edge list
        for each seed.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> verts = np.zeros(3, dtype=np.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg, seeds_offsets = cugraph.induced_subgraph(G, sverts)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G)
    directed = G.is_directed()

    # FIXME: Hardcoded for now
    offsets = None

    if G.renumbered:
        if isinstance(vertices, cudf.DataFrame):
            vertices = G.lookup_internal_vertex_id(vertices, vertices.columns)
        else:
            vertices = G.lookup_internal_vertex_id(vertices)

    vertices = ensure_valid_dtype(G, vertices, "subgraph_vertices")

    if not isinstance(offsets, cudf.Series):
        if isinstance(offsets, list):
            offsets = cudf.Series(offsets)
        elif offsets is None:
            # FIXME: Does the offsets always start from zero?
            offsets = cudf.Series([0, len(vertices)])

    result_graph = Graph(directed=directed)

    do_expensive_check = False
    source, destination, weight, offsets = pylibcugraph_induced_subgraph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        subgraph_vertices=vertices,
        subgraph_offsets=offsets,
        do_expensive_check=do_expensive_check,
    )

    df = cudf.DataFrame()
    df["src"] = source
    df["dst"] = destination
    df["weight"] = weight

    if len(df) == 0:
        return None, None

    seeds_offsets = cudf.Series(offsets)

    if G.renumbered:
        df, src_names = G.unrenumber(df, "src", get_column_names=True)
        df, dst_names = G.unrenumber(df, "dst", get_column_names=True)
    else:
        # FIXME: THe original 'src' and 'dst' are not stored in 'simpleGraph'
        src_names = "src"
        dst_names = "dst"

    if G.edgelist.weights:
        result_graph.from_cudf_edgelist(
            df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        result_graph.from_cudf_edgelist(df, source=src_names, destination=dst_names)

    if isNx is True:
        result_graph = cugraph_to_nx(result_graph)

    return result_graph, seeds_offsets
