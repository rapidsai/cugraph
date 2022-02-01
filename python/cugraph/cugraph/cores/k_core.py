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

from cugraph.cores import k_core_wrapper, core_number_wrapper
from cugraph.utilities import (ensure_cugraph_obj_for_nx,
                               cugraph_to_nx,
                               )
from cugraph.structure.graph_classes import Graph


def k_core(G, k=None, core_number=None):
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

    k : int, optional (default=None)
        Order of the core. This value must not be negative. If set to None, the
        main core is returned.

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
    >>> gdf = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
    ...                     dtype=['int32', 'int32', 'float32'], header=None)
    >>> G = cugraph.Graph()
    >>> G.from_cudf_edgelist(gdf, source='0', destination='1')
    >>> KCoreGraph = cugraph.k_core(G)

    """

    G, isNx = ensure_cugraph_obj_for_nx(G)

    mytype = type(G)
    KCoreGraph = mytype()

    if mytype is not Graph:
        raise Exception("directed graph not supported")

    if core_number is not None:
        if G.renumbered is True:
            if len(G.renumber_map.implementation.col_names) > 1:
                cols = core_number.columns[:-1].to_list()
            else:
                cols = 'vertex'
            core_number = G.add_internal_vertex_id(core_number, 'vertex',
                                                   cols)

    else:
        core_number = core_number_wrapper.core_number(G)
        core_number = core_number.rename(
            columns={"core_number": "values"}, copy=False
        )

    if k is None:
        k = core_number["values"].max()

    k_core_df = k_core_wrapper.k_core(G, k, core_number)

    if G.renumbered:
        k_core_df, src_names = G.unrenumber(k_core_df, "src",
                                            get_column_names=True)
        k_core_df, dst_names = G.unrenumber(k_core_df, "dst",
                                            get_column_names=True)

    if G.edgelist.weights:
        KCoreGraph.from_cudf_edgelist(
            k_core_df, source=src_names, destination=dst_names,
            edge_attr="weight"
        )
    else:
        KCoreGraph.from_cudf_edgelist(
            k_core_df, source=src_names, destination=dst_names,
        )

    if isNx is True:
        KCoreGraph = cugraph_to_nx(KCoreGraph)

    return KCoreGraph
