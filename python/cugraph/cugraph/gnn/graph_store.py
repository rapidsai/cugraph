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
from cugraph.experimental import PropertyGraph
from cugraph.community.egonet import batched_ego_graphs
from cugraph.utilities.utils import sample_groups

import numpy as np
import cupy as cp


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
        return {k: self.gdata._vertex_prop_dataframe[col_names] for k,col_names in self.ndata_key_col_d.items()}

    @property
    def edata(self):
        return {k: self.gdata._edge_prop_dataframe[col_names] for k,col_names in self.edata_key_col_d.items()}

    @property
    def gdata(self):
        return self.__G


    def add_node_data(self, df, node_col_name, node_key, ntype=None):
        """
        Todo: Add docstring
        """
        self.gdata.add_vertex_data(df, vertex_col_name=node_col_name, type_name=ntype)
        col_names = list(df.columns)
        col_names.remove(node_col_name)
        self.ndata_key_col_d[node_key] = col_names

    def add_edge_data(self, df, edge_col_name, edge_key, etype=None):
        """
        Todo: Add docstring
        """
        self.gdata.add_edge_data(data_df, edge_col_name=node_col_name, type_name=etype)
        col_names = list(df.columns)
        col_names.remove(edge_col_name)
        self.edata_key_col_d[edge_key] = col_names

    def get_node_storage(self, key, ntype=None):
        df = self.gdata._vertex_prop_dataframe
        col_names = self.ndata_key_col_d[key]
        return CuFeatureStorage(df=df, type=ntype, col_names=col_names)

    def get_edge_storage(self, key, etype=None):
        col_names = self.edata_key_col_d[key]
        df = self.gdata._edge_prop_dataframe
        return CuFeatureStorage(df=df, type=etype, col_names=col_names)

    def ntypes(self):
        return self.ndata['_TYPE_'].drop_duplicates()

    def __init__(self, graph):
        if isinstance(graph, PropertyGraph):
            self.__G = graph
            #dict to map column names corresponding to edge features of each type
            self.edata_key_col_d = {}
            #dict to map column names corresponding to node features of each type
            self.ndata_key_col_d = {}
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
            If -1 is given all the neighboring edges for each node on
            each edge type will be selected.
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
        CuPy array
            The sampled arrays for bipartite graph.
        """
        num_nodes = len(nodes)
        current_seeds = nodes.reindex(index=np.arange(0, num_nodes))
        _g = self.__G.extract_subgraph(create_using=cugraph.Graph,
                                       allow_multi_edges=True)
        ego_edge_list, seeds_offsets = batched_ego_graphs(_g,
                                                          current_seeds,
                                                          radius=1)
        # filter and get a certain size neighborhood

        # Step 1
        # Get Filtered List of ego_edge_list corresposing to current_seeds
        # We filter by creating a series of destination nodes
        # corresponding to the offsets and filtering non matching vallues

        seeds_offsets_s = cudf.Series(seeds_offsets).values
        offset_lens = seeds_offsets_s[1:] - seeds_offsets_s[0:-1]
        dst_seeds = current_seeds.repeat(offset_lens)
        dst_seeds.index = ego_edge_list.index
        filtered_list = ego_edge_list[ego_edge_list["dst"] == dst_seeds]

        # Step 2
        # Sample Fan Out
        # for each dst take maximum of fanout samples
        filtered_list = sample_groups(filtered_list,
                                      by="dst",
                                      n_samples=fanout)

        return filtered_list['dst'].values, filtered_list['src'].values

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

    def __init__(self, df, type, col_names, backend_lib='torch'):
        self.df = df
        self.type = type
        self.col_names = col_names
        if backend_lib=='torch':
            from torch.utils.dlpack import from_dlpack
        elif backend_lib=='tf':
            from tensorflow.experimental.dlpack import from_dlpack
        else:
            raise NotImplementedError("Only pytorch and tensorflow backends are currently supported")

        self.from_dlpack = from_dlpack


        

    def fetch(self, indices, device, pin_memory=False, **kwargs):
        """ Fetch the features of the given node/edge IDs to the
        given device.

        Parameters
        ----------
        indices : Tensor
            Node or edge IDs.
        device : Device
            Device context.
        pin_memory : 

        Returns
        -------
        Tensor
            Feature data stored in PyTorch Tensor.
        """
        # Default implementation uses synchronous fetch.
        indices = cp.asarray(indices)
        # index first as to avoid transferring the whole frame
        # TODO: verify we set index to ids in property graphs
        subset_df = self.df.loc[indices]
        subset_df = subset_df[subset_df['_TYPE_']==self.type][self.col_names]
        tensor =  self.from_dlpack(subset_df.to_dlpack())

        return tensor.to(device)
