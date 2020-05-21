# Copyright (c) 2019 - 2020, NVIDIA CORPORATION.
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

from cugraph.community import ktruss_subgraph_wrapper
from cugraph.utilities.unrenumber import unrenumber
from cugraph.structure.graph import Graph


def ktruss_subgraph(G, k, use_weights=True):
    """
    Returns the K-Truss subgraph of a graph for a specific k.

    The k-truss of a graph is a subgraph where each edge is part of at least
    (k−2) triangles. K-trusses are used for finding tighlty knit groups of
    vertices in a graph. A k-truss is a relaxation of a k-clique in the graph
    and was define in [1]. Finding cliques is computationally demanding and
    finding the maximal k-clique is known to be NP-Hard.

    In contrast, finding a k-truss is computationally tractable as its
    key building block, namely triangle counting counting, can be executed
    in polnymomial time.Typically, it takes many iterations of triangle
    counting to find the k-truss of a graph. Yet these iterations operate
    on a weakly monotonically shrinking graph.
    Therefore, finding the k-truss of a graph can be done in a fairly
    reasonable amount of time. The solution in cuGraph is based on a
    GPU algorithm first shown in [2] and uses the triangle counting algorithm
    from [3].

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

    k : int
        The desired k to be used for extracting the k-truss subgraph.

    use_weights : Bool
        whether the output should contain the edge weights if G has them

    Returns
    -------
    G_truss : cuGraph.Graph
        A cugraph graph descriptor with the k-truss subgraph for the given k.

    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(M, source='0', destination='1')
    >>> k_subgraph = cugraph.ktruss_subgraph(G, 3)
    """

    KTrussSubgraph = Graph()
    if type(G) is not Graph:
        raise Exception("input graph must be undirected")

    subgraph_df = ktruss_subgraph_wrapper.ktruss_subgraph(G, k, use_weights)
    if G.renumbered:
        subgraph_df = unrenumber(G.edgelist.renumber_map, subgraph_df, 'src')
        subgraph_df = unrenumber(G.edgelist.renumber_map, subgraph_df, 'dst')

    if G.edgelist.weights:
        KTrussSubgraph.from_cudf_edgelist(subgraph_df,
                                          source='src',
                                          destination='dst',
                                          edge_attr='weight')
    else:
        KTrussSubgraph.from_cudf_edgelist(subgraph_df,
                                          source='src',
                                          destination='dst')

    return KTrussSubgraph
