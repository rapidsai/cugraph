# Copyright (c) 2024, NVIDIA CORPORATION.
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

import warnings

from typing import Union, Optional, Dict, Tuple, List

from cugraph.utilities.utils import import_optional
from cugraph.gnn import cugraph_comms_get_raft_handle

import cupy
import pylibcugraph

from cugraph_dgl.typing import TensorType
from cugraph_dgl.utils.cugraph_conversion_utils import _cast_to_torch_tensor
from cugraph_dgl.features import WholeFeatureStore
from cugraph_dgl.view import (
    HeteroNodeView,
    HeteroNodeDataView,
    HeteroEdgeView,
    HeteroEdgeDataView,
    EmbeddingView,
)


# Have to use import_optional even though these are required
# dependencies in order to build properly.
dgl = import_optional("dgl")
torch = import_optional("torch")
tensordict = import_optional("tensordict")

HOMOGENEOUS_NODE_TYPE = "n"
HOMOGENEOUS_EDGE_TYPE = (HOMOGENEOUS_NODE_TYPE, "e", HOMOGENEOUS_NODE_TYPE)


class Graph:
    """
    cuGraph-backed duck-typed version of dgl.DGLGraph that distributes
    the graph across workers.  This object uses lazy graph creation.
    Users can repeatedly call add_edges, and the tensors won't
    be converted into a cuGraph graph until one is needed
    (i.e. when creating a loader). Supports
    single-node/single-GPU, single-node/multi-GPU, and
    multi-node/multi-GPU graph storage.

    Each worker should have a slice of the graph locally, and
    call put_edge_index with its slice.
    """

    def __init__(
        self,
        is_multi_gpu: bool = False,
        ndata_storage="torch",
        edata_storage="torch",
        **kwargs,
    ):
        """
        Parameters
        ----------
        is_multi_gpu: bool (optional, default=False)
            Specifies whether this graph is distributed across GPUs.
        ndata_storage: str (optional, default='torch')
            Specifies where node data should be stored
            (options are 'torch' and 'wholegraph').
            If using PyTorch tensors for storage ('torch')
            then data will be replicated across workers and data
            for all nodes should be provided when calling add_nodes.
            If using WholeGraph wholememory tensors for storage,
            then data will be distributed across workers and only
            the local slice of the data should be provided when
            calling add_nodes.
        edata_storage: str (optional, default='torch')
            If using PyTorch tensors for storage ('torch')
            then data will be replicated across workers and data
            for all nodes should be provided when calling add_edge.
            If using WholeGraph wholememory tensors for storage,
            then data will be distributed across workers and only
            the local slice of the data should be provided when
            calling add_edges.
        kwargs:
            Optional kwargs for WholeGraph feature storage.
        """

        if ndata_storage not in ("torch", "wholegraph"):
            raise ValueError(
                "Invalid node storage type (valid types are 'torch' and 'wholegraph')"
            )
        if edata_storage not in ("torch", "wholegraph"):
            raise ValueError(
                "Invalid edge storage type (valid types are 'torch' and 'wholegraph')"
            )

        self.__num_nodes_dict = {}
        self.__num_edges_dict = {}
        self.__edge_indices = tensordict.TensorDict({}, batch_size=(2,))

        self.__graph = None
        self.__vertex_offsets = None
        self.__handle = None
        self.__is_multi_gpu = is_multi_gpu

        self.__ndata_storage_type = (
            WholeFeatureStore
            if ndata_storage == "wholegraph"
            else dgl.storages.pytorch_tensor.PyTorchTensorStorage
        )
        self.__edata_storage_type = (
            WholeFeatureStore
            if edata_storage == "wholegraph"
            else dgl.storages.pytorch_tensor.PyTorchTensorStorage
        )
        self.__ndata_storage = {}
        self.__edata_storage = {}
        self.__wg_kwargs = kwargs

    @property
    def is_multi_gpu(self):
        return self.__is_multi_gpu

    def to_canonical_etype(
        self, etype: Union[str, Tuple[str, str, str]]
    ) -> Tuple[str, str, str]:
        if etype is None:
            if len(self.canonical_etypes) > 1:
                raise ValueError("Edge type is required for heterogeneous graphs.")
            return HOMOGENEOUS_EDGE_TYPE

        if isinstance(etype, tuple) and len(etype) == 3:
            return etype

        for src_type, rel_type, dst_type in self.__edge_indices.keys(
            leaves_only=True, include_nested=True
        ):
            if etype == rel_type:
                return (src_type, rel_type, dst_type)

        raise ValueError("Unknown relation type " + etype)

    def add_nodes(
        self,
        global_num_nodes: int,
        data: Optional[Dict[str, TensorType]] = None,
        ntype: Optional[str] = None,
    ):
        """
        Adds the given number of nodes to this graph.  Can only be called once
        per node type. The number of nodes specified here refers to the total
        number of nodes across all workers (the entire graph). If the backing
        feature store is distributed (i.e. wholegraph), then only local features
        should be passed to the data argument.  If the backing feature store is
        replicated, then features for all nodes in the graph should be passed to
        the data argument, including those for nodes not on the local worker.

        Parameters
        ----------
        global_num_nodes: int
            The total number of nodes of the given type in this graph.
            The same number should be passed to every worker.
        data: Dict[str, TensorType] (optional, default=None)
            Node feature tensors.
        ntype: str (optional, default=None)
            The node type being modified.  Required for heterogeneous graphs.
        """
        if ntype is None:
            if len(self.__num_nodes_dict.keys()) > 1:
                raise ValueError("Node type is required for heterogeneous graphs.")
            ntype = HOMOGENEOUS_NODE_TYPE

        if ntype in self.__num_nodes_dict:
            raise ValueError(
                "Calling add_nodes multiple types for the same "
                "node type is not allowed in cuGraph-DGL"
            )

        if self.is_multi_gpu:
            # Ensure all nodes got the same number of nodes passed
            world_size = torch.distributed.get_world_size()
            local_size = torch.tensor(
                [global_num_nodes], device="cuda", dtype=torch.int64
            )
            ns = torch.empty((world_size,), device="cuda", dtype=torch.int64)
            torch.distributed.all_gather_into_tensor(ns, local_size)
            if not (ns == global_num_nodes).all():
                raise ValueError("The global number of nodes must match on all workers")

            # Ensure the sum of the feature shapes equals the global number of nodes.
            if data is not None:
                for feature_name, feature_tensor in data.items():
                    features_size = torch.tensor(
                        [int(feature_tensor.shape[0])], device="cuda", dtype=torch.int64
                    )
                    torch.distributed.all_reduce(
                        features_size, op=torch.distributed.ReduceOp.SUM
                    )
                    if features_size != global_num_nodes:
                        raise ValueError(
                            "The total length of the feature vector across workers must"
                            " match the global number of nodes but it does not "
                            f"match for {feature_name}."
                        )

        self.__num_nodes_dict[ntype] = global_num_nodes

        if data is not None:
            for feature_name, feature_tensor in data.items():
                self.__ndata_storage[ntype, feature_name] = self.__ndata_storage_type(
                    _cast_to_torch_tensor(feature_tensor), **self.__wg_kwargs
                )

        self.__graph = None
        self.__vertex_offsets = None

    def __check_node_ids(self, ntype: str, ids: TensorType):
        """
        Ensures all node ids in the provided id tensor are valid.
        Raises a ValueError if any are invalid.

        Parameters
        ----------
        ntype: str
            The node type being validated against.
        ids:
            The tensor of ids being validated.
        """
        if ntype in self.__num_nodes_dict:
            if ids.max() + 1 > self.num_nodes(ntype):
                raise ValueError(
                    f"input tensor contains invalid node ids for type {ntype}"
                )
        else:
            raise ValueError(
                f"add_nodes() must be called for type {ntype} before calling num_edges."
            )

    def add_edges(
        self,
        u: TensorType,
        v: TensorType,
        data: Optional[Dict[str, TensorType]] = None,
        etype: Optional[Union[str, Tuple[str, str, str]]] = None,
    ) -> None:
        """
        Adds edges to this graph.  Must be called after add_nodes
        is called for the src/dst node type. If the backing feature
        store is distributed (i.e. wholegraph), then only local
        features should be passed to the data argument.  If the
        backing feature store is replicated, then features for
        all edges should be passed to the data argument,
        including those for edges not on the local worker.

        Parameters
        ----------
        u: TensorType
            1d tensor of source node ids (local slice of the distributed edgelist).
        v: TensorType
            1d tensor of destination node ids (local slice of the distributed edgelist).
        data: Dict[str, TensorType] (optional, default=None)
            Dictionary containing edge features for the new edges.
        etype: Union[str, Tuple[str, str, str]]
            The edge type of the edges being inserted.  Not required
            for homogeneous graphs, which have only one edge type.
        """

        # Validate all inputs before proceeding
        # The number of nodes for the src/dst type needs to be known and there cannot
        # be any edges of this type in the graph.
        dgl_can_edge_type = self.to_canonical_etype(etype)
        src_type, _, dst_type = dgl_can_edge_type
        if dgl_can_edge_type in self.__edge_indices.keys(
            leaves_only=True, include_nested=True
        ):
            raise ValueError(
                "This cuGraph-DGL graph already contains edges of type"
                f" {dgl_can_edge_type}. Calling add_edges multiple times"
                " for the same edge type is not supported."
            )
        self.__check_node_ids(src_type, u)
        self.__check_node_ids(dst_type, v)

        self.__edge_indices[dgl_can_edge_type] = torch.stack(
            [
                _cast_to_torch_tensor(u),
                _cast_to_torch_tensor(v),
            ]
        ).to(self.idtype)

        if data is not None:
            for attr_name, attr_tensor in data.items():
                self.__edata_storage[
                    dgl_can_edge_type, attr_name
                ] = self.__edata_storage_type(
                    _cast_to_torch_tensor(attr_tensor), **self.__wg_kwargs
                )

        num_edges = self.__edge_indices[dgl_can_edge_type].shape[1]
        if self.is_multi_gpu:
            num_edges = torch.tensor([num_edges], device="cuda", dtype=torch.int64)
            torch.distributed.all_reduce(num_edges, op=torch.distributed.ReduceOp.SUM)

        self.__num_edges_dict[dgl_can_edge_type] = int(num_edges)

        self.__graph = None
        self.__vertex_offsets = None

    def num_nodes(self, ntype: Optional[str] = None) -> int:
        """
        Returns the number of nodes of ntype, or if ntype is not provided,
        the total number of nodes in the graph.
        """
        if ntype is None:
            return sum(self.__num_nodes_dict.values())

        return self.__num_nodes_dict[ntype]

    def number_of_nodes(self, ntype: Optional[str] = None) -> int:
        """
        Alias for num_nodes.
        """
        return self.num_nodes(ntype=ntype)

    def num_edges(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Returns the number of edges of etype, or if etype is not provided,
        the total number of edges in the graph.
        """
        if etype is None:
            return sum(self.__num_edges_dict.values())

        etype = self.to_canonical_etype(etype)
        return self.__num_edges_dict[etype]

    def number_of_edges(self, etype: Union[str, Tuple[str, str, str]] = None) -> int:
        """
        Alias for num_edges.
        """
        return self.num_edges(etype=etype)

    @property
    def ntypes(self) -> List[str]:
        """
        Returns the node type names in this graph.
        """
        return list(self.__num_nodes_dict.keys())

    @property
    def etypes(self) -> List[str]:
        """
        Returns the edge type names in this graph
        (the second element of the canonical edge
        type tuple).
        """
        return [et[1] for et in self.__num_edges_dict.keys()]

    @property
    def canonical_etypes(self) -> List[str]:
        """
        Returns the canonical edge type names in this
        graph.
        """
        return list(self.__num_edges_dict.keys())

    @property
    def _vertex_offsets(self) -> Dict[str, int]:
        if self.__vertex_offsets is None:
            ordered_keys = sorted(list(self.ntypes))
            self.__vertex_offsets = {}
            offset = 0
            for vtype in ordered_keys:
                self.__vertex_offsets[vtype] = offset
                offset += self.num_nodes(vtype)

        return dict(self.__vertex_offsets)

    def __get_edgelist(self, prob_attr=None) -> Dict[str, "torch.Tensor"]:
        """
        This function always returns src/dst labels with respect
        to the out direction.

        Returns
        -------
        Dict[str, torch.Tensor] with the following keys:
            src: source vertices (int64)
                Note that src is the 1st element of the DGL edge index.
            dst: destination vertices (int64)
                Note that dst is the 2nd element of the DGL edge index.
            eid: edge ids for each edge (int64)
                Note that these start from 0 for each edge type.
            etp: edge types for each edge (int32)
                Note that these are in lexicographic order.
        """
        sorted_keys = sorted(
            list(self.__edge_indices.keys(leaves_only=True, include_nested=True))
        )

        # note that this still follows the DGL convention of (src, rel, dst)
        # i.e. (author, writes, paper): [[0,1,2],[2,0,1]] is referring to a
        # cuGraph graph where (paper 2) -> (author 0), (paper 0) -> (author 1),
        # and (paper 1) -> (author 0)
        edge_index = torch.concat(
            [
                torch.stack(
                    [
                        self.__edge_indices[src_type, rel_type, dst_type][0]
                        + self._vertex_offsets[src_type],
                        self.__edge_indices[src_type, rel_type, dst_type][1]
                        + self._vertex_offsets[dst_type],
                    ]
                )
                for (src_type, rel_type, dst_type) in sorted_keys
            ],
            axis=1,
        ).cuda()

        edge_type_array = torch.arange(
            len(sorted_keys), dtype=torch.int32, device="cuda"
        ).repeat_interleave(
            torch.tensor(
                [self.__edge_indices[et].shape[1] for et in sorted_keys],
                device="cuda",
                dtype=torch.int32,
            )
        )

        num_edges_t = torch.tensor(
            [self.__edge_indices[et].shape[1] for et in sorted_keys], device="cuda"
        )

        if self.is_multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            num_edges_all_t = torch.empty(
                world_size, num_edges_t.numel(), dtype=torch.int64, device="cuda"
            )
            torch.distributed.all_gather_into_tensor(num_edges_all_t, num_edges_t)

            start_offsets = num_edges_all_t[:rank].T.sum(axis=1)

        else:
            rank = 0
            start_offsets = torch.zeros(
                (len(sorted_keys),), dtype=torch.int64, device="cuda"
            )
            num_edges_all_t = num_edges_t.reshape((1, num_edges_t.numel()))

        # Use pinned memory here for fast access to CPU/WG storage
        edge_id_array_per_type = [
            torch.arange(
                start_offsets[i],
                start_offsets[i] + num_edges_all_t[rank][i],
                dtype=torch.int64,
                device="cpu",
            ).pin_memory()
            for i in range(len(sorted_keys))
        ]

        # Retrieve the weights from the appropriate feature(s)
        # DGL implicitly requires all edge types use the same
        # feature name.
        if prob_attr is None:
            weights = None
        else:
            if len(sorted_keys) > 1:
                weights = torch.concat(
                    [
                        self.edata[prob_attr][sorted_keys[i]][ix]
                        for i, ix in enumerate(edge_id_array_per_type)
                    ]
                )
            else:
                weights = self.edata[prob_attr][edge_id_array_per_type[0]]

        # Safe to move this to cuda because the consumer will always
        # move it to cuda if it isn't already there.
        edge_id_array = torch.concat(edge_id_array_per_type).cuda()

        edgelist_dict = {
            "src": edge_index[0],
            "dst": edge_index[1],
            "etp": edge_type_array,
            "eid": edge_id_array,
        }

        if weights is not None:
            edgelist_dict["wgt"] = weights

        return edgelist_dict

    @property
    def is_homogeneous(self):
        return len(self.__num_edges_dict) <= 1 and len(self.__num_nodes_dict) <= 1

    @property
    def idtype(self):
        return torch.int64

    @property
    def _resource_handle(self):
        if self.__handle is None:
            if self.is_multi_gpu:
                self.__handle = pylibcugraph.ResourceHandle(
                    cugraph_comms_get_raft_handle().getHandle()
                )
            else:
                self.__handle = pylibcugraph.ResourceHandle()
        return self.__handle

    def _graph(
        self,
        direction: str,
        prob_attr: Optional[str] = None,
    ) -> Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph]:
        """
        Gets the pylibcugraph Graph object with edges pointing in the given direction
        (i.e. 'out' is standard, 'in' is reverse).
        """

        if direction not in ["out", "in"]:
            raise ValueError(f"Invalid direction {direction} (expected 'in' or 'out').")

        graph_properties = pylibcugraph.GraphProperties(
            is_multigraph=True, is_symmetric=False
        )

        if self.__graph is not None:
            if (
                self.__graph["direction"] != direction
                or self.__graph["prob_attr"] != prob_attr
            ):
                self.__graph = None

        if self.__graph is None:
            src_col, dst_col = ("src", "dst") if direction == "out" else ("dst", "src")
            edgelist_dict = self.__get_edgelist(prob_attr=prob_attr)

            if self.is_multi_gpu:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

                vertices_array = cupy.arange(self.num_nodes(), dtype="int64")
                vertices_array = cupy.array_split(vertices_array, world_size)[rank]

                graph = pylibcugraph.MGGraph(
                    self._resource_handle,
                    graph_properties,
                    [cupy.asarray(edgelist_dict[src_col]).astype("int64")],
                    [cupy.asarray(edgelist_dict[dst_col]).astype("int64")],
                    vertices_array=[vertices_array],
                    edge_id_array=[cupy.asarray(edgelist_dict["eid"])],
                    edge_type_array=[cupy.asarray(edgelist_dict["etp"])],
                    weight_array=[cupy.asarray(edgelist_dict["wgt"])]
                    if "wgt" in edgelist_dict
                    else None,
                )
            else:
                graph = pylibcugraph.SGGraph(
                    self._resource_handle,
                    graph_properties,
                    cupy.asarray(edgelist_dict[src_col]).astype("int64"),
                    cupy.asarray(edgelist_dict[dst_col]).astype("int64"),
                    vertices_array=cupy.arange(self.num_nodes(), dtype="int64"),
                    edge_id_array=cupy.asarray(edgelist_dict["eid"]),
                    edge_type_array=cupy.asarray(edgelist_dict["etp"]),
                    weight_array=cupy.asarray(edgelist_dict["wgt"])
                    if "wgt" in edgelist_dict
                    else None,
                )

        self.__graph = {"graph": graph, "direction": direction, "prob_attr": prob_attr}

        return self.__graph["graph"]

    def _has_n_emb(self, ntype: str, emb_name: str) -> bool:
        return (ntype, emb_name) in self.__ndata_storage

    def _get_n_emb(
        self, ntype: Union[str, None], emb_name: str, u: Union[str, TensorType]
    ) -> Union["torch.Tensor", "EmbeddingView"]:
        """
        Gets the embedding of a single node type.
        Unlike DGL, this function takes the string node
        type name instead of an integer id.

        Parameters
        ----------
        ntype: str
            The node type to get the embedding of.
        emb_name: str
            The embedding name of the embedding to get.
        u: Union[str, TensorType]
            Nodes to get the representation of, or ALL
            to get the representation of all nodes of
            the given type (returns embedding view).

        Returns
        -------
        Union[torch.Tensor, cugraph_dgl.view.EmbeddingView]
            The embedding of the given edge type with the given embedding name.
        """

        if ntype is None:
            if len(self.ntypes) == 1:
                ntype = HOMOGENEOUS_NODE_TYPE
            else:
                raise ValueError("Must provide the node type for a heterogeneous graph")

        if dgl.base.is_all(u):
            return EmbeddingView(
                self.__ndata_storage[ntype, emb_name], self.num_nodes(ntype)
            )

        try:
            return self.__ndata_storage[ntype, emb_name].fetch(
                _cast_to_torch_tensor(u), "cuda"
            )
        except RuntimeError as ex:
            warnings.warn(
                "Got error accessing data, trying again with index on device: "
                + str(ex)
            )
            return self.__ndata_storage[ntype, emb_name].fetch(
                _cast_to_torch_tensor(u).cuda(), "cuda"
            )

    def _has_e_emb(self, etype: Tuple[str, str, str], emb_name: str) -> bool:
        return (etype, emb_name) in self.__edata_storage

    def _get_e_emb(
        self, etype: Tuple[str, str, str], emb_name: str, u: Union[str, TensorType]
    ) -> "torch.Tensor":
        """
        Gets the embedding of a single edge type.
        Unlike DGL, this function takes the canonical edge type
        instead of an integer id.

        Parameters
        ----------
        etype: str
            The edge type to get the embedding of.
        emb_name: str
            The embedding name of the embedding to get.
        u: Union[str, TensorType]
            Edges to get the representation of, or ALL to
            get the representation of all nodes of the
            given type.

        Returns
        -------
        torch.Tensor
            The embedding of the given edge type with the given embedding name.
        """

        etype = self.to_canonical_etype(etype)

        if dgl.base.is_all(u):
            return EmbeddingView(
                self.__edata_storage[etype, emb_name], self.num_edges(etype)
            )

        try:
            return self.__edata_storage[etype, emb_name].fetch(
                _cast_to_torch_tensor(u), "cuda"
            )
        except RuntimeError as ex:
            warnings.warn(
                "Got error accessing data, trying again with index on device: "
                + str(ex)
            )
            return self.__edata_storage[etype, emb_name].fetch(
                _cast_to_torch_tensor(u).cuda(), "cuda"
            )

    def _set_n_emb(
        self, ntype: str, u: Union[str, TensorType], kv: Dict[str, TensorType]
    ) -> None:
        """
        Stores or updates the embedding(s) of a single node type.
        Unlike DGL, this function takes the string node type name
        instead of an integer id.

        The semantics of this function match those of add_nodes
        with respect to whether or not the backing feature store
        is distributed.

        Parameters
        ----------
        ntype: str
            The node type to store an embedding of.
        u: Union[str, TensorType]
            The indices to update, if updating the embedding.
            Currently, updating a slice of an embedding is
            unsupported, so this should be ALL.
        kv: Dict[str, TensorType]
            A mapping of embedding names to embedding tensors.
        """

        if not dgl.base.is_all(u):
            raise NotImplementedError(
                "Updating a slice of an embedding is "
                "currently unimplemented in cuGraph-DGL."
            )

        for k, v in kv:
            self.__ndata_storage[ntype, k] = self.__ndata_storage_type(
                v,
                **self.__wg_kwargs,
            )

    def _set_e_emb(
        self, etype: str, u: Union[str, TensorType], kv: Dict[str, TensorType]
    ) -> None:
        """
        Stores or updates the embedding(s) of a single edge type.
        Unlike DGL, this function takes the canonical edge type name
        instead of an integer id.

        The semantics of this function match those of add_edges
        with respect to whether or not the backing feature store
        is distributed.

        Parameters
        ----------
        etype: str
            The edge type to store an embedding of.
        u: Union[str, TensorType]
            The indices to update, if updating the embedding.
            Currently, updating a slice of an embedding is
            unsupported, so this should be ALL.
        kv: Dict[str, TensorType]
            A mapping of embedding names to embedding tensors.
        """

        if not dgl.base.is_all(u):
            raise NotImplementedError(
                "Updating a slice of an embedding is "
                "currently unimplemented in cuGraph-DGL."
            )

        for k, v in kv:
            self.__edata_storage[etype, k] = self.__edata_storage_type(
                v,
                **self.__wg_kwargs,
            )

    def _pop_n_emb(self, ntype: str, key: str) -> "torch.Tensor":
        """
        Removes and returns the embedding of the given node
        type with the given name.

        Parameters
        ----------
        ntype:str
            The node type.
        key:str
            The embedding name.

        Returns
        -------
        The removed embedding.
        """
        return self.__ndata_storage[ntype, key].pop(key)

    def _pop_e_emb(self, etype: str, key: str) -> "torch.Tensor":
        """
        Removes and returns the embedding of the given edge
        type with the given name.

        Parameters
        ----------
        etype:str
            The node type.
        key:str
            The embedding name.

        Returns
        -------
        torch.Tensor
            The removed embedding.
        """
        return self.__edata_storage[etype, key].pop(key)

    def _get_n_emb_keys(self, ntype: str) -> List[str]:
        """
        Gets a list of the embedding names for a given node
        type.

        Parameters
        ----------
        ntype: str
            The node type to get embedding names for.

        Returns
        -------
        List[str]
            The list of embedding names for the given node type.
        """
        return [k for (t, k) in self.__ndata_storage if ntype == t]

    def _get_e_emb_keys(self, etype: str) -> List[str]:
        """
        Gets a list of the embedding names for a given edge
        type.

        Parameters
        ----------
        etype: str
            The edge type to get embedding names for.

        Returns
        -------
        List[str]
            The list of embedding names for the given edge type.
        """
        return [k for (t, k) in self.__edata_storage if etype == t]

    def all_edges(
        self,
        form="uv",
        order="eid",
        etype: Union[str, Tuple[str, str, str]] = None,
        device: Union[str, int, "torch.device"] = "cpu",
    ):
        """
        Returns all edges with the specified edge type.
        cuGraph-DGL currently only supports 'eid' format and
        'eid' order.

        Parameters
        ----------
        form: str (optional, default='uv')
            The format to return ('uv', 'eid', 'all').

        order: str (optional, default='eid')
            The order to return edges in ('eid', 'srcdst')
            cuGraph-DGL currently only supports 'eid'.
        etype: Union[str, Tuple[str, str, str]] (optional, default=None)
            The edge type to get.  Not required if this is
            a homogeneous graph.  Can be the relation type if the
            relation type is unique, or the canonical edge type.
        device: Union[str, int, torch.device] (optional, default='cpu')
            The device where returned edges should be stored
            ('cpu', 'cuda', or device id).
        """

        if order != "eid":
            raise NotImplementedError("cugraph-DGL only supports eid order.")

        if etype is None and len(self.canonical_etypes) > 1:
            raise ValueError("Edge type is required for heterogeneous graphs.")

        etype = self.to_canonical_etype(etype)

        if form == "eid":
            return torch.arange(
                0,
                self.__num_edges_dict[etype],
                dtype=self.idtype,
                device=device,
            )
        else:
            if self.is_multi_gpu:
                # This can't be done because it requires collective communication.
                raise ValueError(
                    "Calling all_edges in a distributed graph with"
                    " form 'uv' or 'all' is unsupported."
                )

            else:
                eix = self.__edge_indices[etype].to(device)
                if form == "uv":
                    return eix[0], eix[1]
                elif form == "all":
                    return (
                        eix[0],
                        eix[1],
                        torch.arange(
                            self.__num_edges_dict[etype],
                            dtype=self.idtype,
                            device=device,
                        ),
                    )
                else:
                    raise ValueError(f"Invalid form {form}")

    @property
    def ndata(self) -> HeteroNodeDataView:
        """
        Returns a view of the node data in this graph which can be used to
        access or modify node features.
        """

        if len(self.ntypes) == 1:
            ntype = self.ntypes[0]
            return HeteroNodeDataView(self, ntype, dgl.base.ALL)

        return HeteroNodeDataView(self, self.ntypes, dgl.base.ALL)

    @property
    def edata(self) -> HeteroEdgeDataView:
        """
        Returns a view of the edge data in this graph which can be used to
        access or modify edge features.
        """
        if len(self.canonical_etypes) == 1:
            return HeteroEdgeDataView(self, None, dgl.base.ALL)

        return HeteroEdgeDataView(self, self.canonical_etypes, dgl.base.ALL)

    @property
    def nodes(self) -> HeteroNodeView:
        """
        Returns a view of the nodes in this graph.
        """
        return HeteroNodeView(self)

    @property
    def edges(self) -> HeteroEdgeView:
        """
        Returns a view of the edges in this graph.
        """
        return HeteroEdgeView(self)
