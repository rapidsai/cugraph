# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.tree import minimum_spanning_tree_wrapper
from cugraph.structure.graph import Graph
from cugraph.utilities import check_nx_graph
from cugraph.utilities import cugraph_to_nx


def minimum_spanning_tree(
    G, weight=None, algorithm="boruvka", ignore_nan=False
):
    """
    Returns a minimum spanning tree (MST) or forest (MSF) on an undirected graph

    Parameters
    ----------
    G : cuGraph.Graph or networkx.Graph
        cuGraph graph descriptor with connectivity information.
    weight : string
        default to the weights in the graph, if the graph edges do not have a weight attribute a default weight of 1 will be used.
    algorithm : string
        Default to 'boruvka'. The parallel algorithm to use when finding a minimum spanning tree. The .
    ignore_nan : bool
        Default to False
    Returns
    -------
    G_mst : cuGraph.Graph or networkx.Graph
        A cugraph graph descriptor with a minimum spanning tree or forest.
        The networkx graph will not have all attributes copied over
    """

    G, isNx = check_nx_graph(G)

    if isNx is True:
        mst = minimum_spanning_tree(G)
        return cugraph_to_nx(mst)
    else:
        return minimum_spanning_tree(G)
