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

import cudf

from pylibcugraph import (
    core_number as pylibcugraph_core_number,
    k_core as pylibcugraph_k_core,
    ResourceHandle,
)

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    cugraph_to_nx,
)


def _call_plc_core_number(G, degree_type):
    vertex, core_number = pylibcugraph_core_number(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        degree_type=degree_type,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["core_number"] = core_number
    return df


def k_core(G, k=None, core_number=None, degree_type="bidirectional"):
    """
    Compute the k-core of the graph G based on the out degree of its nodes. A
    k-core of a graph is a maximal subgraph that contains nodes of degree k or
    more. This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information. The graph
        should contain undirected edges where undirected edges are represented
        as directed edges in both directions. While this graph can contain edge
        weights, they don't participate in the calculation of the k-core.
        The current implementation only supports undirected graphs.

    k : int, optional (default=None)
        Order of the core. This value must not be negative. If set to None, the
        main core is returned.

    degree_type: str, (default="bidirectional")
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.

    core_number : cudf.DataFrame, optional (default=None)
        Precomputed core number of the nodes of the graph G containing two
        cudf.Series of size V: the vertex identifiers and the corresponding
        core number values. If set to None, the core numbers of the nodes are
        calculated internally.

        core_number['vertex'] : cudf.Series
            Contains the vertex identifiers
        core_number['values'] : cudf.Series
            Contains the core number of vertices

    Returns
    -------
    KCoreGraph : cuGraph.Graph
        K Core of the input graph

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> KCoreGraph = cugraph.k_core(G)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if degree_type not in ["incoming", "outgoing", "bidirectional"]:
        raise ValueError(
            f"'degree_type' must be either incoming, "
            f"outgoing or bidirectional, got: {degree_type}"
        )

    mytype = type(G)

    KCoreGraph = mytype()

    if G.is_directed():
        raise ValueError("G must be an undirected Graph instance")

    if core_number is None:
        core_number = _call_plc_core_number(G, degree_type=degree_type)
    else:
        if G.renumbered:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = core_number.columns[:-1].to_list()
            else:
                cols = "vertex"

            core_number = G.add_internal_vertex_id(core_number, "vertex", cols)

    core_number = core_number.rename(columns={"core_number": "values"})
    if k is None:
        k = core_number["values"].max()

    src_vertices, dst_vertices, weights = pylibcugraph_k_core(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        degree_type=degree_type,
        k=k,
        core_result=core_number,
        do_expensive_check=False,
    )

    k_core_df = cudf.DataFrame()
    k_core_df["src"] = src_vertices
    k_core_df["dst"] = dst_vertices
    k_core_df["weight"] = weights

    if G.renumbered:
        k_core_df, src_names = G.unrenumber(k_core_df, "src", get_column_names=True)
        k_core_df, dst_names = G.unrenumber(k_core_df, "dst", get_column_names=True)

    else:
        src_names = k_core_df.columns[0]
        dst_names = k_core_df.columns[1]

    if G.edgelist.weights:

        KCoreGraph.from_cudf_edgelist(
            k_core_df, source=src_names, destination=dst_names, edge_attr="weight"
        )
    else:
        KCoreGraph.from_cudf_edgelist(
            k_core_df,
            source=src_names,
            destination=dst_names,
        )

    if isNx is True:
        KCoreGraph = cugraph_to_nx(KCoreGraph)

    return KCoreGraph
