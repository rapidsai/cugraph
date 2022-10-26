# Copyright (c) 2022, NVIDIA CORPORATION.
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


import cugraph
import cudf


def node_subgraph(
    pg,
    nodes=None,
    create_using=cugraph.MultiGraph,
):
    """
    Return a subgraph induced on the given nodes.

    A node-induced subgraph is a graph with edges whose endpoints are both
    in the specified node set.

    Parameters
    ----------
    pg: Property Graph
        The graph to create subgraph from
    nodes : Tensor
        The nodes to form the subgraph.
    Returns
    -------
    cuGraph
        The sampled subgraph with the same node ID space with the original
        graph.
    """

    _g = pg.extract_subgraph(create_using=create_using, check_multi_edges=True)

    if nodes is None:
        return _g
    else:
        _n = cudf.Series(nodes)
        _subg = cugraph.subgraph(_g, _n)
        return _subg
