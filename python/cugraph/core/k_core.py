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

from cugraph.core import k_core_wrapper
from cugraph.structure.graph import Graph


def k_core(G,
           k=None,
           core_number=None):
    """
    Compute the core numbers for the nodes of the graph G.

    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed or undirected edges where undirected edges are
        represented as directed edges in both directions.
    k : int, optional
        Order of the core. If set to None, the main core is returned.
    core_number : cudf.DataFrame, optional
        Precomputer core number of the nodes of the graph G containing two
        cudf.Series of size V: the vertex identifiers and the corresponding
        core number values.

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
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> KCoreGraph = cugraph.core_number(G)
    """

    KCoreGraph = Graph()
    k_core_wrapper.k_core(G.graph_ptr, k, core_number, KCoreGraph.graph_ptr)

    return KCoreGraph
