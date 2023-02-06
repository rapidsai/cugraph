# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Union
from functools import cached_property
from cugraph.utilities.utils import import_optional, MissingModule
from cugraph.gnn import FeatureStore
from cugraph.gnn.dgl_extensions.dgl_uniform_sampler import DGLUniformSampler
import cudf
import dask_cudf
import cupy as cp
from cugraph_dgl.utils.cugraph_storage_utils import (
    _assert_valid_canonical_etype,
    backend_dtype_to_np_dtype_dict,
    add_edge_ids_to_edges_dict,
    add_node_offset_to_edges_dict,
)
from cugraph_dgl.utils.feature_storage import dgl_FeatureStorage

dgl = import_optional("dgl")
F = import_optional("dgl.backend")
torch = import_optional("torch")


class CuGraphStorage:
    """
    Duck-typed version of the DGLHeteroGraph class made for cuGraph
    for storing graph structure and node/edge feature data.

    This object is wrapper around cugraph's Multi GPU MultiGraph and returns samples
    that conform with `DGLHeteroGraph`
    See: https://docs.rapids.ai/api/cugraph/nightly/api_docs/cugraph_dgl.html
    """

    def __init__(
        self,
        data_dict: Dict[
            Tuple[str, str, str], Union[cudf.DataFrame, dask_cudf.DataFrame]
        ],
        num_nodes_dict: Dict[str, int],
        single_gpu: bool = True,
        device_id: int = 0,
        idtype=None if isinstance(F, MissingModule) else F.int64,
    ):
        """
        Constructor for creating a object of instance CuGraphStorage

        See also ``cugraph_dgl.cugraph_storage_from_heterograph``
        to convert from DGLHeteroGraph to CuGraphStorage

        Parameters
        ----------
         data_dict:
            The dictionary data for constructing a heterogeneous graph.
            The keys are in the form of string triplets (src_type, edge_type, dst_type),
            specifying the source node, edge, and destination node types.
            The values are graph data is a dataframe with 2 columns form of (𝑈,𝑉),
            where (𝑈[𝑖],𝑉[𝑖]) forms the edge with ID 𝑖.
         num_nodes_dict: dict[str, int]
            The number of nodes for some node types, which is a
            dictionary mapping a node type T to the number of T-typed nodes.
        single_gpu: bool
            Whether to create the cugraph Property Graph
            on a single GPU or multiple GPUs
            single GPU = True
            single GPU = False
        device_id: int
            If specified, must be the integer ID of the GPU device to have the
            results being created on
        idtype: Framework-specific device object,
            The data type for storing the structure-related graph
            information this can be ``torch.int32`` or ``torch.int64``
            for PyTorch.
            Defaults to ``torch.int64`` if pytorch is installed
         Examples
         --------
         The following example uses `CuGraphStorage` :
            >>> from cugraph_dgl.cugraph_storage import CuGraphStorage
            >>> import cudf
            >>> import torch
            >>> num_nodes_dict={"drug": 3, "gene": 2, "disease": 1}
            >>> drug_interacts_drug_df = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
            >>> drug_interacts_gene = cudf.DataFrame({"src": [0, 1], "dst": [0, 1]})
            >>> drug_treats_disease = cudf.DataFrame({"src": [1], "dst": [0]})
            >>> data_dict = {("drug", "interacts", "drug"):drug_interacts_drug_df,
                 ("drug", "interacts", "gene"):drug_interacts_gene,
                 ("drug", "treats", "disease"):drug_treats_disease }
            >>> gs = CuGraphStorage(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
            >>> gs.add_node_data(ntype='drug', feat_name='node_feat',
                                          feat_obj=torch.as_tensor([0.1, 0.2, 0.3]))
            >>> gs.add_edge_data(canonical_etype=("drug", "interacts", "drug"),
                                          feat_name='edge_feat',
                                          feat_obj=torch.as_tensor([0.2, 0.4]))
            >>> gs.ntypes
            ['disease', 'drug', 'gene']
            >>> gs.etypes
            ['interacts', 'interacts', 'treats']
            >>> gs.canonical_etypes
            [('drug', 'interacts', 'drug'),
             ('drug', 'interacts', 'gene'),
             ('drug', 'treats', 'disease')]

            >>> gs.sample_neighbors({'disease':[0]},
                                    1)
            Graph(num_nodes={'disease': 1, 'drug': 3, 'gene': 2},
            num_edges={('drug', 'interacts', 'drug'): 0,
                       ('drug', 'interacts', 'gene'): 0,
                       ('drug', 'treats', 'disease'): 1},
            metagraph=[('drug', 'drug', 'interacts'),
                       ('drug', 'gene', 'interacts'),
                       ('drug', 'disease', 'treats')])

            >>> gs.get_node_storage(key='node_feat',
                                    ntype='drug').fetch([0,1,2])
            tensor([0.1000, 0.2000, 0.3000], device='cuda:0',
             dtype=torch.float64)

            >>> es = gs.get_edge_storage(key='edge_feat',
                                    etype=('drug', 'interacts', 'drug'))
            >>> es.fetch([0,1])
            tensor([0.2000, 0.4000], device='cuda:0', dtype=torch.float64)
        """
        # Order is very important
        # do this first before cuda work
        # Create cuda context on the right gpu,
        # defaults to gpu-0
        import numba.cuda as cuda

        cuda.select_device(device_id)

        self.idtype = idtype
        self.id_np_type = backend_dtype_to_np_dtype_dict[idtype]
        self.num_nodes_dict = num_nodes_dict
        self._ntype_offset_d = self.__get_ntype_offset_d(self.num_nodes_dict)
        # Todo: Can possibly optimize by persisting edge-list
        # Trade-off memory for run-time
        self.num_edges_dict = {k: len(v) for k, v in data_dict.items()}
        self._etype_offset_d = self.__get_etype_offset_d(self.num_edges_dict)
        self.single_gpu = single_gpu

        self.ndata_storage = FeatureStore(backend="torch")
        self.ndata = self.ndata_storage.fd
        self.edata_storage = FeatureStore(backend="torch")
        self.edata = self.edata_storage.fd

        self._etype_range_d = self.__get_etype_range_d(
            self._etype_offset_d, self.num_canonical_edges_dict
        )
        _edges_dict = add_edge_ids_to_edges_dict(
            data_dict, self._etype_offset_d, self.id_np_type
        )

        self._edges_dict = add_node_offset_to_edges_dict(
            _edges_dict, self._ntype_offset_d
        )
        self._etype_id_dict = {
            etype: etype_id for etype_id, etype in enumerate(self.canonical_etypes)
        }
        self.uniform_sampler = None

    def add_node_data(self, feat_obj: Sequence, ntype: str, feat_name: str):
        """
        Add node features

        Parameters
        ----------
        df : array_like object
            The node feature to save in feature store
        ntype : str
            The node type to be added.
            For example, if dataframe contains data about users, ntype
            might be "users".
        feat_name : str
            The name of the feature being stored
        Returns
        -------
        None
        """
        self.ndata_storage.add_data(
            feat_obj=feat_obj,
            type_name=ntype,
            feat_name=feat_name,
        )

    def add_edge_data(
        self,
        feat_obj: Sequence,
        canonical_etype: Tuple[str, str, str],
        feat_name: str,
    ):
        """
        Add edge features

        Parameters
        ----------
        feat_obj : array_like object
            The edge feature to save in feature store
        canonical_etype : Tuple[(str, str, str)]
            The edge type to be added
        feat_name : string
        Returns
        -------
        None
        """
        _assert_valid_canonical_etype(canonical_etype)
        self.edata_storage.add_data(
            feat_obj=feat_obj,
            type_name=canonical_etype,
            feat_name=feat_name,
        )

    # Sampling Function
    def sample_neighbors(
        self,
        nodes,
        fanout: int,
        edge_dir: str = "in",
        prob: Optional[str] = None,
        exclude_edges=None,
        replace: bool = False,
        output_device=None,
    ):
        """
        Return a DGLGraph which is a subgraph induced by sampling neighboring
        edges of the given nodes.
        See ``dgl.sampling.sample_neighbors`` for detailed semantics.
        Parameters
        ----------
        nodes : Tensor or dict[str, Tensor]
            Node IDs to sample neighbors from.
            This argument can take a single ID tensor or a dictionary of node
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        fanout : int or dict[etype, int]
            The number of edges to be sampled for each node on each edge type.
            This argument can take a single int or a dictionary of edge types
            and ints. If a single int is given, DGL will sample this number of
            edges for each node for every edge type.
            If -1 is given for a single edge type, all the neighboring edges
            with that edge type will be selected.
        edge_dir: 'in' or 'out'
            The direction of edges to import
        prob : str, optional
            Feature name used as the (un-normalized) probabilities associated
            with each neighboring edge of a node.  The feature must have only
            one element for each edge.
            The features must be non-negative floats, and the sum of the
            features of inbound/outbound edges for every node must be positive
            (though they don't have to sum up to one).  Otherwise, the result
            will be undefined. If :attr:`prob` is not None, GPU sampling is
            not supported.
        exclude_edges: tensor or dict
            Edge IDs to exclude during sampling neighbors for the seed nodes.
            This argument can take a single ID tensor or a dictionary of edge
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        replace : bool, optional
            If True, sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            A sampled subgraph with the same nodes as the original graph, but
            only the sampled neighboring edges.  The induced edge IDs will be
            in ``edata[dgl.EID]``.
        """
        if self.uniform_sampler is None:
            self.uniform_sampler = DGLUniformSampler(
                self._edges_dict,
                self._etype_range_d,
                self._etype_id_dict,
                self.single_gpu,
            )

        if prob is not None:
            raise NotImplementedError(
                "prob is not currently supported",
                " for sample_neighbors in CuGraphStorage",
            )

        if exclude_edges is not None:
            raise NotImplementedError(
                "exclude_edges is not currently supported",
                " for sample_neighbors in CuGraphStorage",
            )

        if not isinstance(nodes, dict):
            if len(self.ntypes) > 1:
                raise dgl.DGLError(
                    "Must specify node type when the graph is not homogeneous."
                )
            nodes = cp.asarray(nodes)
            nodes = {self.ntypes[0]: nodes}
        else:
            nodes = {
                k: self.dgl_n_id_to_cugraph_id(F.tensor(n), k) for k, n in nodes.items()
            }
            nodes = {k: cp.asarray(F.tensor(n)) for k, n in nodes.items()}

        sampled_obj = self.uniform_sampler.sample_neighbors(
            nodes,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
        )
        # heterograph case
        if len(self.etypes) > 1:
            graph_data_d, graph_eid_d = self.__convert_to_dgl_tensor_d(
                sampled_obj, self.idtype
            )
            sampled_graph = dgl.heterograph(
                data_dict=graph_data_d,
                num_nodes_dict=self.num_nodes_dict,
                idtype=self.idtype,
            )
            sampled_graph.edata[dgl.EID] = graph_eid_d
        else:
            src_ids, dst_ids, edge_ids = sampled_obj
            src_ids = torch.as_tensor(src_ids, device="cuda")
            dst_ids = torch.as_tensor(dst_ids, device="cuda")
            edge_ids = torch.as_tensor(edge_ids, device="cuda")
            total_number_of_nodes = self.total_number_of_nodes
            sampled_graph = dgl.graph(
                (src_ids, dst_ids),
                num_nodes=total_number_of_nodes,
                idtype=self.idtype,
            )
            sampled_graph.edata[dgl.EID] = edge_ids

        # to device function move the dgl graph to desired devices
        if output_device is not None:
            sampled_graph.to(output_device)
        return sampled_graph

    # Required in Cluster-GCN
    def subgraph(self, nodes, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given nodes.
        This has the same semantics as ``dgl.node_subgraph``.
        Parameters
        ----------
        nodes : nodes or dict[str, nodes]
            The nodes to form the subgraph. The allowed nodes formats are:
            * Int Tensor: Each element is a node ID. The tensor must have the
             same device type and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether node :math:`i` is in the subgraph.
             If the graph is homogeneous, directly pass the above formats.
             Otherwise, the argument must be a dictionary with keys being
             node types and values being the node IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError("subgraph is not implemented yet")

    # Required in Link Prediction
    # relabel = F we use dgl functions,
    # relabel = T, we need to delete nodes and relabel
    def edge_subgraph(self, edges, relabel_nodes=False, output_device=None):
        """
        Return a subgraph induced on given edges.
        This has the same semantics as ``dgl.edge_subgraph``.
        Parameters
        ----------
        edges : edges or dict[(str, str, str), edges]
            The edges to form the subgraph. The allowed edges formats are:
            * Int Tensor: Each element is an edge ID. The tensor must have the
              same device type and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether edge :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above
            formats. Otherwise, the argument must be a dictionary with keys
            being edge types and values being the edge IDs in the above formats
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError("edge_subgraph is not implemented yet")

    # Required in Link Prediction negative sampler
    def find_edges(
        self, eid, etype: Optional[Tuple[str, str, str]] = None, output_device=None
    ):
        """
        Return the source and destination node ID(s) given the edge ID(s).

        Parameters
        ----------
        eid : edge ID(s)
            The edge IDs. The allowed formats are:

            * ``int``: A single ID.
            * Int Tensor: Each element is an ID.
            The tensor must have the same device type
            and ID data type as the graph's.
            * iterable[int]: Each element is an ID.

        etype : Tuple[str, str, str]
            The type name of the edges.
            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor
            The source node IDs of the edges.
            The i-th element is the source node ID of the i-th edge.
        Tensor
            The destination node IDs of the edges.
            The i-th element is the destination node ID of the i-th edge.
        """

        if etype:
            src_type, connection_type, dst_type = etype
        eid = self.dgl_e_id_to_cugraph_id(eid, etype)
        # TODO: implement below
        src, dst = self.find_edges(eid, etype)
        src = torch.as_tensor(src, device="cuda")
        dst = torch.as_tensor(dst, device="cuda")
        src = self.cugraph_n_id_to_dgl_id(src, src_type)
        dst = self.cugraph_n_id_to_dgl_id(dst, dst_type)

        return src, dst

    # Required in Link Prediction negative sampler
    def global_uniform_negative_sampling(
        self, num_samples, exclude_self_loops=True, replace=False, etype=None
    ):
        """
        Per source negative sampling as in ``dgl.dataloading.GlobalUniform``
        """
        raise NotImplementedError(
            "global_uniform_negative_sampling not implemented yet"
        )

    def get_node_storage(self, key: str, ntype: str = None):
        """
        Get storage object of node feature of
        type :attr:`ntype` and name :attr:`key`
        """
        if ntype is None:
            if len(self.ntypes) > 1:
                raise ValueError(
                    "ntype must be provided if multiple ntypes are present in the graph"
                )
            else:
                ntype = self.ntype[0]
        return dgl_FeatureStorage(self.ndata_storage, type_name=ntype, feat_name=key)

    def get_edge_storage(self, key: str, etype: Optional[Tuple[str, str, str]] = None):
        """
        Get storage object of edge feature of
        type :attr:`ntype` and name :attr:`key`
        """
        if etype is None:
            if len(self.etypes) > 1:
                raise ValueError(
                    "etype must be provided if multiple etypes are present in the graph"
                )
            else:
                etype = self.etypes[0]
        return dgl_FeatureStorage(self.edata_storage, type_name=etype, feat_name=key)

    # Number of edges/nodes utils
    def num_nodes(self, ntype: str = None) -> int:
        """
        Return the number of nodes in the graph.
        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type.
            If not given (default), it  returns the total number of nodes
            of all types.

        Returns
        -------
        int
            The number of nodes.
        """
        if ntype:
            return self.num_nodes_dict[ntype]
        else:
            return self.total_number_of_nodes

    def number_of_nodes(self, ntype: str = None) -> int:
        """
        Return the number of nodes in the graph.
        Alias of ``num_nodes``
        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type.
            If not given (default), it  returns the total number of nodes
            of all types.

        Returns
        -------
        int
            The number of nodes.
        """
        return self.num_nodes(ntype)

    @property
    def ntypes(self) -> Sequence[str]:
        """
        Return all the node type names in the graph.

        Returns
        -------
        list[str]
            All the node type names in a list.
        """
        ntypes = list(self.num_nodes_dict.keys())
        return ntypes

    @property
    def etypes(self) -> Sequence[str]:
        """
        Return all the edge type names in the graph.

        Returns
        -------
        list[str]
            All the edge type names in a list.
        """

        return [can_etype[1] for can_etype in self.canonical_etypes]

    def num_edges(self, etype: Optional[str] = None) -> int:
        """
        Return the number of edges in the graph.
        Parameters
        ----------
        etype:

        Returns
        -------
        int
            The number of edges
        """
        if etype:
            if etype not in self.canonical_etypes:
                etype = self.get_corresponding_canonical_etype(etype)
            return self.num_edges_dict[etype]
        else:
            return self.total_number_of_edges

    @cached_property
    def total_number_of_edges(self) -> int:
        return sum(self.num_edges_dict.values())

    @cached_property
    def total_number_of_nodes(self) -> int:
        return sum(self.num_nodes_dict.values())

    @property
    def num_canonical_edges_dict(self) -> dict[str, int]:
        return self.num_edges_dict

    @property
    def canonical_etypes(self) -> Sequence[Tuple[str, str, str]]:
        return list(self.num_edges_dict.keys())

    @property
    def device(self):
        """
        Get the device of the graph.
        Returns
        -------
        device context
            The device of the graph, which should be a
            framework-specific device object (e.g., ``torch.device``).
        """
        return torch.cuda.current_device()

    # Index Conversion Utils
    def get_node_id_offset(self, ntype: str) -> int:
        """
        Return the integer offset for node id of type ntype
        """
        return self._ntype_offset_d[ntype]

    def get_edge_id_offset(self, canonical_etype: Tuple[str, str, str]) -> int:
        """
        Return the integer offset for node id of type etype
        """
        _assert_valid_canonical_etype(canonical_etype)
        return self._etype_offset_d[canonical_etype]

    def dgl_n_id_to_cugraph_id(self, index_t, ntype: str):
        return index_t + self.get_node_id_offset(ntype)

    def cugraph_n_id_to_dgl_id(self, index_t, ntype: str):
        return index_t - self.get_node_id_offset(ntype)

    def dgl_e_id_to_cugraph_id(self, index_t, canonical_etype: Tuple[str, str, str]):
        return index_t + self.get_edge_id_offset(canonical_etype)

    def cugraph_e_id_to_dgl_id(self, index_t, canonical_etype: Tuple[str, str, str]):
        return index_t - self.get_edge_id_offset(canonical_etype)

    # Methods for getting the offsets per type
    @staticmethod
    def __get_etype_offset_d(num_canonical_edges_dict):
        last_st = 0
        etype_st_d = {}
        for etype in sorted(num_canonical_edges_dict.keys()):
            etype_st_d[etype] = last_st
            last_st = last_st + num_canonical_edges_dict[etype]
        return etype_st_d

    @staticmethod
    def __get_etype_range_d(etype_offset_d, num_canonical_edges_dict):
        # dict for edge_id_offset_start
        etype_range_d = {}
        for etype, st in etype_offset_d.items():
            etype_range_d[etype] = (st, st + num_canonical_edges_dict[etype])
        return etype_range_d

    @staticmethod
    def __get_ntype_offset_d(num_nodes_dict):
        # dict for node_id_offset_start
        last_st = 0
        ntype_st_d = {}
        for ntype in sorted(num_nodes_dict.keys()):
            ntype_st_d[ntype] = last_st
            last_st = last_st + num_nodes_dict[ntype]
        return ntype_st_d

    def get_corresponding_canonical_etype(self, etype: str) -> str:
        can_etypes = [
            can_etype for can_etype in self.canonical_etypes if can_etype[1] == etype
        ]
        if len(can_etypes) > 1:
            raise dgl.DGLError(
                f'Edge type "{etype}" is ambiguous. Please use canonical'
                + "edge type in the form of (srctype, etype, dsttype)"
            )
        return can_etypes[0]

    def __convert_to_dgl_tensor_d(
        self,
        graph_sampled_data_d,
        o_dtype=None if isinstance(F, MissingModule) else F.int64,
    ):

        graph_data_d = {}
        graph_eid_d = {}
        for canonical_etype, (
            src,
            dst,
            edge_id,
        ) in graph_sampled_data_d.items():
            src_type = canonical_etype[0]
            dst_type = canonical_etype[2]

            src_t = _torch_tensor_from_cp_array(src)
            dst_t = _torch_tensor_from_cp_array(dst)
            edge_id_t = _torch_tensor_from_cp_array(edge_id)

            src_t = self.cugraph_n_id_to_dgl_id(src_t, src_type)
            dst_t = self.cugraph_n_id_to_dgl_id(dst_t, dst_type)
            edge_id_t = self.cugraph_e_id_to_dgl_id(edge_id_t, canonical_etype)
            graph_data_d[canonical_etype] = (src_t.to(o_dtype), dst_t.to(o_dtype))
            graph_eid_d[canonical_etype] = edge_id_t.to(o_dtype)

        return graph_data_d, graph_eid_d


def _torch_tensor_from_cp_array(ar):
    if len(ar) == 0:
        return torch.as_tensor(ar.get()).to("cuda")
    return torch.as_tensor(ar, device="cuda")
