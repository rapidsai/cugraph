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

from typing import Union

import cudf
from pylibcugraph import k_truss_subgraph as pylibcugraph_k_truss_subgraph
from pylibcugraph import ResourceHandle
from cugraph.structure.graph_classes import Graph
from cugraph.utilities import (
    ensure_cugraph_obj_for_nx,
    cugraph_to_nx,
)
from cugraph.utilities.utils import import_optional

# FIXME: the networkx.Graph type used in the type annotation for
# ktruss_subgraph() is specified using a string literal to avoid depending on
# and importing networkx. Instead, networkx is imported optionally, which may
# cause a problem for a type checker if run in an environment where networkx is
# not installed.
networkx = import_optional("networkx")


def k_truss(
    G: Union[Graph, "networkx.Graph"], k: int
) -> Union[Graph, "networkx.Graph"]:
    """
    Returns the K-Truss subgraph of a graph for a specific k.

    The k-truss of a graph is a subgraph where each edge is incident to at
    least (k−2) triangles. K-trusses are used for finding tighlty knit groups
    of vertices in a graph. A k-truss is a relaxation of a k-clique in the graph.
    Finding cliques is computationally demanding and finding the maximal
    k-clique is known to be NP-Hard.

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information. k-Trusses are
        defined for only undirected graphs as they are defined for
        undirected triangle in a graph.

    k : int
        The desired k to be used for extracting the k-truss subgraph.

    Returns
    -------
    G_truss : cuGraph.Graph or networkx.Graph
        A cugraph graph descriptor with the k-truss subgraph for the given k.
        The networkx graph will NOT have all attributes copied over

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> k_subgraph = cugraph.k_truss(G, 3)

    """
    G, isNx = ensure_cugraph_obj_for_nx(G)

    if isNx is True:
        k_sub = ktruss_subgraph(G, k, use_weights=False)
        S = cugraph_to_nx(k_sub)
        return S
    else:
        return ktruss_subgraph(G, k, use_weights=False)


# FIXME: merge this function with k_truss


def ktruss_subgraph(
    G: Union[Graph, "networkx.Graph"],
    k: int,
    use_weights=True,  # deprecated
) -> Graph:
    """
    Returns the K-Truss subgraph of a graph for a specific k.

    NOTE: this function is currently not available on CUDA 11.4 systems.

    The k-truss of a graph is a subgraph where each edge is part of at least
    (k−2) triangles. K-trusses are used for finding tighlty knit groups of
    vertices in a graph. A k-truss is a relaxation of a k-clique in the graph
    and was define in [1]. Finding cliques is computationally demanding and
    finding the maximal k-clique is known to be NP-Hard.

    In contrast, finding a k-truss is computationally tractable as its
    key building block, namely triangle counting, can be executed
    in polnymomial time.Typically, it takes many iterations of triangle
    counting to find the k-truss of a graph. Yet these iterations operate
    on a weakly monotonically shrinking graph.
    Therefore, finding the k-truss of a graph can be done in a fairly
    reasonable amount of time. The solution in cuGraph is based on a
    GPU algorithm first shown in [2] and uses the triangle counting algorithm
    from [3].

    References
    ----------

    [1] Cohen, J.,
    "Trusses: Cohesive subgraphs for social network analysis"
    National security agency technical report, 2008

    [2] O. Green, J. Fox, E. Kim, F. Busato, et al.
    “Quickly Finding a Truss in a Haystack”
    IEEE High Performance Extreme Computing Conference (HPEC), 2017
    https://doi.org/10.1109/HPEC.2017.8091038

    [3] O. Green, P. Yalamanchili, L.M. Munguia,
    “Fast Triangle Counting on GPU”
    Irregular Applications: Architectures and Algorithms (IA3), 2014


    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. k-Trusses are
        defined for only undirected graphs as they are defined for
        undirected triangle in a graph.
        The current implementation only supports undirected graphs.

    k : int
        The desired k to be used for extracting the k-truss subgraph.

    Returns
    -------
    G_truss : cuGraph.Graph
        A cugraph graph descriptor with the k-truss subgraph for the given k.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> k_subgraph = cugraph.ktruss_subgraph(G, 3, use_weights=False)
    """

    KTrussSubgraph = Graph()
    if G.is_directed():
        raise ValueError("input graph must be undirected")

    sources, destinations, edge_weights, _ = pylibcugraph_k_truss_subgraph(
        resource_handle=ResourceHandle(),
        graph=G._plc_graph,
        k=k,
        do_expensive_check=True,
    )

    subgraph_df = cudf.DataFrame()
    subgraph_df["src"] = sources
    subgraph_df["dst"] = destinations
    if edge_weights is not None:
        subgraph_df["weight"] = edge_weights

    if G.renumbered:
        subgraph_df = G.unrenumber(subgraph_df, "src")
        subgraph_df = G.unrenumber(subgraph_df, "dst")

    if G.edgelist.weights:
        KTrussSubgraph.from_cudf_edgelist(
            subgraph_df, source="src", destination="dst", edge_attr="weight"
        )
    else:
        KTrussSubgraph.from_cudf_edgelist(subgraph_df, source="src", destination="dst")

    return KTrussSubgraph
