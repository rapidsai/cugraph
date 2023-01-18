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

from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    df_score_to_dictionary,
)
import cudf

from pylibcugraph import core_number as pylibcugraph_core_number, ResourceHandle


def core_number(G, degree_type="bidirectional"):
    """
    Compute the core numbers for the nodes of the graph G. A k-core of a graph
    is a maximal subgraph that contains nodes of degree k or more.
    A node has a core number of k if it belongs a k-core but not to k+1-core.
    This call does not support a graph with self-loops and parallel
    edges.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        The graph should contain undirected edges where undirected edges are
        represented as directed edges in both directions. While this graph
        can contain edge weights, they don't participate in the calculation
        of the core numbers.
        The current implementation only supports undirected graphs.

    degree_type: str, (default="bidirectional")
        This option determines if the core number computation should be based
        on input, output, or both directed edges, with valid values being
        "incoming", "outgoing", and "bidirectional" respectively.

    Returns
    -------
    df : cudf.DataFrame or python dictionary (in NetworkX input)
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding core number values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['core_number'] : cudf.Series
            Contains the core number of vertices

    Examples
    --------
    >>> from cugraph.experimental.datasets import karate
    >>> G = karate.get_graph(fetch=True)
    >>> df = cugraph.core_number(G)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G)

    if G.is_directed():
        raise ValueError("input graph must be undirected")

    if degree_type not in ["incoming", "outgoing", "bidirectional"]:
        raise ValueError(
            f"'degree_type' must be either incoming, "
            f"outgoing or bidirectional, got: {degree_type}"
        )

    vertex, core_number = pylibcugraph_core_number(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        degree_type=degree_type,
        do_expensive_check=False,
    )

    df = cudf.DataFrame()
    df["vertex"] = vertex
    df["core_number"] = core_number

    if G.renumbered:
        df = G.unrenumber(df, "vertex")

    if isNx is True:
        df = df_score_to_dictionary(df, "core_number")

    return df
