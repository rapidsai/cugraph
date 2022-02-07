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

import cudf
import cugraph
from cugraph.experimental import EXPERIMENTAL__PropertyGraph as PropertyGraph
from cugraph.community.egonet import batched_ego_graphs


class CuGraphStore:
    """
    A wrapper around a cuGraph Property Graph that
    then adds functions to basically match the DGL GraphStorage API.
    This is not a full duck-types match to a DGL GraphStore.  This class
    return cuGraph types and had additional functional arguments.
    For true integration with DGL, a second class would need to be written
    in DGL that handles the conversion to other types, like DGLGraph, and
    handles the extra arguments.

    homogeneous graphs, graphs with no attributes - use Property Graph
    hetrogeneous graphs - use PropertyGraph
    """

    @property
    def ndata(self):
        raise NotImplementedError("not yet implemented")

    @property
    def edata(self):
        raise NotImplementedError("not yet implemented")

    @property
    def gdata(self):
        return self.__G

    def __init__(self, graph):
        if isinstance(graph, PropertyGraph):
            self.__G = graph
        else:
            raise ValueError("graph must be a PropertyGraph")

    ######################################
    # Utilities
    ######################################
    @property
    def num_vertices(self):
        return self.__G.num_vertices

    @property
    def num_edges(self):
        return self.__G.num_edges

    def get_vertex_ids(self):
        return self.__G.vertices_ids()

    ######################################
    # Sampling APIs
    ######################################

    def sample_neighbors(self,
                         nodes,
                         fanout=-1,
                         edge_dir='in',
                         prob=None,
                         replace=False):
        """
        Sample neighboring edges of the given nodes and return the subgraph.

        Parameters
        ----------
        nodes : array (single dimension)
            Node IDs to sample neighbors from.
        fanout : int
            The number of edges to be sampled for each node on each edge type.
        edge_dir : str {"in" or "out"}
            Determines whether to sample inbound or outbound edges.
            Can take either in for inbound edges or out for outbound edges.
        prob : str
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node. Each feature must be a
            scalar. The features must be non-negative floats, and the sum of
            the features of inbound/outbound edges for every node must be
            positive (though they don't have to sum up to one). Otherwise,
            the result will be undefined. If not specified, sample uniformly.
        replace : bool
            If True, sample with replacement.

        Returns
        -------
        DGLGraph
            The sampled subgraph with the same node ID space with the original
            graph.
        """
        pass

    def node_subgraph(self,
                      nodes=None,
                      create_using=cugraph.Graph,
                      directed=False,
                      multigraph=True):
        """
        Return a subgraph induced on the given nodes.

        A node-induced subgraph is a graph with edges whose endpoints are both
        in the specified node set.

        Parameters
        ----------
        nodes : Tensor
            The nodes to form the subgraph.

        Returns
        -------
        cuGraph
            The sampled subgraph with the same node ID space with the original
            graph.
        """

        # expr="(_SRC in nodes) | (_DST_ in nodes)"

        _g = self.__G.extract_subgraph(
                        create_using=cugraph.Graph(directed=directed),
                        allow_multi_edges=multigraph)

        if nodes is None:
            return _g
        else:
            _n = cudf.Series(nodes)
            _subg = cugraph.subgraph(_g, _n)
            return _subg

    def egonet(self, nodes, k):
        """Return the k-hop egonet of the given nodes.

        A k-hop egonet of a node is the subgraph induced by the k-hop neighbors
        of the node.

        Parameters
        ----------
        nodes : single dimension array
            The center nodes of the egonets.

        Returns
        -------
        ego_edge_lists :  cudf.DataFrame
            GPU data frame containing all induced sources identifiers,
            destination identifiers, edge weights

        seeds_offsets: cudf.Series
            Series containing the starting offset in the returned edge list
            for each seed.
        """

        _g = self.__G.extract_subgraph(create_using=cugraph.Graph,
                                       allow_multi_edges=True)

        ego_edge_list, seeds_offsets = batched_ego_graphs(_g, nodes, radius=k)

        return ego_edge_list, seeds_offsets

    def randomwalk(self,
                   nodes,
                   length,
                   prob=None,
                   restart_prob=None):
        """
        Perform randomwalks starting from the given nodes and return the
        traces.

        A k-hop egonet of a node is the subgraph induced by the k-hop
        neighbors of the node.

        Parameters
        ----------
        nodes : single dimension array
            The nodes to start the walk.
        length : int
            Walk length.
        prob : str
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node. Each feature must be a
            scalar.
            The features must be non-negative floats, and the sum of the
            features of inbound/outbound edges for every node must be positive
            (though they don't have to sum up to one). Otherwise, the result
            will be undefined. If not specified, pick the next stop uniformly.
        restart_prob : float
            Probability to terminate the current trace before each transition.

        Returns
        -------
        traces : Tensor
            A 2-D tensor of shape (len(nodes), length + 1). traces[i] stores
            the node IDs reached by the randomwalk starting from nodes[i]. -1
            means the walk has stopped.
        """
        _g = self.__G.extract_subgraph(create_using=cugraph.Graph,
                                       allow_multi_edges=True)

        p, w, s = cugraph.random_walks(_g, nodes,
                                       max_depth=length, use_padding=True)

        return p, w, s


class CuFeatureStorage:
    """Storage for node/edge feature data.

    Either subclassing this class or implementing the same set of interfaces
    is fine. DGL simply uses duck-typing to implement its sampling pipeline.
    """

    def __getitem__(self, ids):
        """Fetch the features of the given node/edge IDs.

        Parameters
        ----------
        ids : Tensor
            Node or edge IDs.

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        pass

    async def async_fetch(self, ids, device):
        """Asynchronously fetch the features of the given node/edge IDs to the
        given device.

        Parameters
        ----------
        ids : Tensor
            Node or edge IDs.
        device : Device
            Device context.

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        # Default implementation uses synchronous fetch.
        return self.__getitem__(ids).to(device)
