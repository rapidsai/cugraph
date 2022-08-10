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

from collections import defaultdict
import cudf
import cugraph
from cugraph.experimental import PropertyGraph
from cugraph.community.egonet import batched_ego_graphs
from cugraph.sampling import uniform_neighbor_sample
import cupy as cp
from functools import cached_property


src_n = PropertyGraph.src_col_name
dst_n = PropertyGraph.dst_col_name
type_n = PropertyGraph.type_col_name
eid_n = PropertyGraph.edge_id_col_name
vid_n = PropertyGraph.vertex_col_name


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

    def __init__(self, graph, backend_lib="torch"):
        if isinstance(graph, PropertyGraph):
            self.__G = graph
        else:
            raise ValueError("graph must be a PropertyGraph")
        # dict to map column names corresponding to edge features
        # of each type
        self.edata_key_col_d = defaultdict(list)
        # dict to map column names corresponding to node features
        # of each type
        self.ndata_key_col_d = defaultdict(list)
        self.backend_lib = backend_lib

    def add_node_data(self, df, node_col_name, node_key, ntype=None):
        self.gdata.add_vertex_data(
            df, vertex_col_name=node_col_name, type_name=ntype
        )
        col_names = list(df.columns)
        col_names.remove(node_col_name)
        self.ndata_key_col_d[node_key] += col_names

    def add_edge_data(self, df, vertex_col_names, edge_key, etype=None):
        self.gdata.add_edge_data(
            df, vertex_col_names=vertex_col_names, type_name=etype
        )
        col_names = [
            col for col in list(df.columns) if col not in vertex_col_names
        ]
        self.edata_key_col_d[edge_key] += col_names

    def get_node_storage(self, key, ntype=None):

        if ntype is None:
            ntypes = self.ntypes
            if len(self.ntypes) > 1:
                raise ValueError(
                    (
                        "Node type name must be specified if there "
                        "are more than one node types."
                    )
                )
            ntype = ntypes[0]
        # FIXME: Remove once below lands
        # https://github.com/rapidsai/cugraph/pull/2444
        df = self.gdata._vertex_prop_dataframe
        col_names = self.ndata_key_col_d[key]
        return CuFeatureStorage(
            df=df,
            id_col=vid_n,
            _type_=ntype,
            col_names=col_names,
            backend_lib=self.backend_lib,
        )

    def get_edge_storage(self, key, etype=None):
        if etype is None:
            etypes = self.etypes
            if len(self.etypes) > 1:
                raise ValueError(
                    (
                        "Edge type name must be specified if there"
                        "are more than one edge types."
                    )
                )

            etype = etypes[0]
        col_names = self.edata_key_col_d[key]
        # FIXME: Remove once below lands
        # https://github.com/rapidsai/cugraph/pull/2444
        df = self.gdata._edge_prop_dataframe
        return CuFeatureStorage(
            df=df,
            id_col=eid_n,
            _type_=etype,
            col_names=col_names,
            backend_lib=self.backend_lib,
        )

    def num_nodes(self, ntype=None):
        return self.gdata.get_num_vertices(ntype)

    def num_edges(self, etype=None):
        return self.gdata.get_num_edges(etype)

    @property
    def ntypes(self):
        # FIXME: Remove once below is fixed
        # https://github.com/rapidsai/cugraph/issues/2423
        s = self.gdata._vertex_prop_dataframe[type_n]
        ntypes = s.drop_duplicates().to_arrow().to_pylist()
        return ntypes

    @property
    def etypes(self):
        # FIXME: Remove once below is fixed
        # https://github.com/rapidsai/cugraph/issues/2423
        s = self.gdata._edge_prop_dataframe[type_n]
        ntypes = s.drop_duplicates().to_arrow().to_pylist()
        return ntypes

    @property
    def ndata(self):
        return {
            # FIXME: Remove once below lands
            # https://github.com/rapidsai/cugraph/pull/2444
            k: self.gdata._vertex_prop_dataframe[col_names].dropna(how="all")
            for k, col_names in self.ndata_key_col_d.items()
        }

    @property
    def edata(self):
        return {
            # FIXME: Remove once below lands
            # https://github.com/rapidsai/cugraph/pull/2444
            k: self.gdata._edge_prop_dataframe[col_names].dropna(how="all")
            for k, col_names in self.edata_key_col_d.items()
        }

    @property
    def gdata(self):
        return self.__G

    ######################################
    # Utilities
    ######################################
    @property
    def num_vertices(self):
        return self.gdata.get_num_vertices()

    def get_vertex_ids(self):
        return self.gdata.vertices_ids()

    ######################################
    # Sampling APIs
    ######################################

    def sample_neighbors(
        self, nodes, fanout=-1, edge_dir="in", prob=None, replace=False
    ):
        """
        Sample neighboring edges of the given nodes and return the subgraph.

        Parameters
        ----------
        nodes_cap : Dlpack of Node IDs (single dimension)
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
        DLPack capsule
            The src nodes for the sampled bipartite graph.
        DLPack capsule
            The sampled dst nodes for the sampledbipartite graph.
        DLPack capsule
            The corresponding eids for the sampled bipartite graph
        """

        if edge_dir not in ["in", "out"]:
            raise ValueError(
                f"edge_dir must be either 'in' or 'out' got {edge_dir} instead"
            )

        if edge_dir == "in":
            sg = self.extracted_reverse_subgraph_without_renumbering
        else:
            sg = self.extracted_subgraph_without_renumbering

        if not hasattr(self, '_sg_node_dtype'):
            self._sg_node_dtype = sg.edgelist.edgelist_df['src'].dtype

        # Uniform sampling assumes fails when the dtype
        # if the seed dtype is not same as the node dtype
        nodes = cudf.from_dlpack(nodes).astype(self._sg_node_dtype)

        sampled_df = uniform_neighbor_sample(
            sg, start_list=nodes, fanout_vals=[fanout],
            with_replacement=replace,
            is_edge_ids=True
            # FIXME: is_edge_ids=True does not seem to do anything
            # issue https://github.com/rapidsai/cugraph/issues/2562
        )

        # handle empty graph case
        if len(sampled_df) == 0:
            return None, None, None

        # we reverse directions when directions=='in'
        if edge_dir == "in":
            sampled_df.rename(
                columns={"destinations": src_n, "sources": dst_n}, inplace=True
            )
        else:
            sampled_df.rename(
                columns={"sources": src_n, "destinations": dst_n}, inplace=True
            )

        return (
            sampled_df[src_n].to_dlpack(),
            sampled_df[dst_n].to_dlpack(),
            sampled_df['indices'].to_dlpack(),
        )

    @cached_property
    def extracted_reverse_subgraph_without_renumbering(self):
        # TODO: Switch to extract_subgraph based on response on
        # https://github.com/rapidsai/cugraph/issues/2458
        subset_df = self.gdata._edge_prop_dataframe[[src_n, dst_n, eid_n]]
        subset_df.rename(columns={src_n: dst_n, dst_n: src_n}, inplace=True)
        subgraph = cugraph.Graph(directed=True)
        subgraph.from_cudf_edgelist(
            subset_df,
            source=src_n,
            destination=dst_n,
            edge_attr=eid_n,
            renumber=False,
            legacy_renum_only=False,
        )
        return subgraph

    @cached_property
    def extracted_subgraph_without_renumbering(self):
        gr_template = cugraph.Graph(directed=True)
        subgraph = self.gdata.extract_subgraph(create_using=gr_template,
                                               edge_weight_property=eid_n,
                                               renumber_graph=False)
        return subgraph

    def find_edges(self, edge_ids_cap, etype):
        """Return the source and destination node IDs given the edge IDs within
        the given edge type.

        Parameters
        ----------
        edge_ids_cap :  Dlpack of Node IDs (single dimension)
            The edge ids  to find

        Returns
        -------
        DLPack capsule
            The src nodes for the given ids

        DLPack capsule
            The dst nodes for the given ids
        """
        edge_ids = cudf.from_dlpack(edge_ids_cap)

        # FIXME: Remove once below lands
        # https://github.com/rapidsai/cugraph/issues/2444
        edge_df = self.gdata._edge_prop_dataframe[[src_n, dst_n,
                                                   eid_n, type_n]]

        subset_df = get_subset_df(
            edge_df, PropertyGraph.edge_id_col_name, edge_ids, etype
        )
        return subset_df[src_n].to_dlpack(), subset_df[dst_n].to_dlpack()

    def node_subgraph(
        self,
        nodes=None,
        create_using=cugraph.Graph,
        directed=False,
        multigraph=True,
    ):
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
        # Values vary b/w cugraph and DGL investigate
        # expr="(_SRC in nodes) | (_DST_ in nodes)"
        _g = self.__G.extract_subgraph(
            create_using=cugraph.Graph(directed=directed),
            allow_multi_edges=multigraph,
        )

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

        _g = self.__G.extract_subgraph(
            create_using=cugraph.Graph, allow_multi_edges=True
        )

        ego_edge_list, seeds_offsets = batched_ego_graphs(_g, nodes, radius=k)

        return ego_edge_list, seeds_offsets

    def randomwalk(self, nodes, length, prob=None, restart_prob=None):
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
        _g = self.__G.extract_subgraph(
            create_using=cugraph.Graph, allow_multi_edges=True
        )

        p, w, s = cugraph.random_walks(
            _g, nodes, max_depth=length, use_padding=True
        )

        return p, w, s


class CuFeatureStorage:
    """Storage for node/edge feature data.

    Either subclassing this class or implementing the same set of interfaces
    is fine. DGL simply uses duck-typing to implement its sampling pipeline.
    """

    def __init__(self, df, id_col, _type_, col_names, backend_lib="torch"):
        self.df = df
        self.id_col = id_col
        self.type = _type_
        self.col_names = col_names
        if backend_lib == "torch":
            from torch.utils.dlpack import from_dlpack
        elif backend_lib == "tf":
            from tensorflow.experimental.dlpack import from_dlpack
        elif backend_lib == "cupy":
            from cupy import from_dlpack
        else:
            raise NotImplementedError(
                "Only pytorch and tensorflow backends are currently supported"
            )

        self.from_dlpack = from_dlpack

    def fetch(self, indices, device, pin_memory=False, **kwargs):
        """Fetch the features of the given node/edge IDs to the
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

        subset_cols = self.col_names + [type_n, self.id_col]
        subset_df = get_subset_df(
            self.df[subset_cols], self.id_col, indices, self.type
        )[self.col_names]
        tensor = self.from_dlpack(subset_df.to_dlpack())

        if isinstance(tensor, cp.ndarray):
            # can not transfer to
            # a different device for cupy
            return tensor
        else:
            return tensor.to(device)


def get_subset_df(df, id_col, indices, _type_):
    """
    Util to get the subset dataframe to the indices of the requested type
    """
    # We can avoid all of this if we set index to id_col like
    # edge_id_col_name and vertex_id_col_name and make it much faster
    # by using loc
    indices_df = cudf.Series(cp.asarray(indices), name=id_col).to_frame()
    id_col_name = id_col + "_index_"
    indices_df = indices_df.reset_index(drop=False).rename(
        columns={"index": id_col_name}
    )
    subset_df = indices_df.merge(df, how="left")
    if _type_ is None:
        subset_df = subset_df[subset_df[type_n].isnull()]
    else:
        subset_df = subset_df[subset_df[type_n] == _type_]
    subset_df = subset_df.sort_values(by=id_col_name)
    return subset_df
