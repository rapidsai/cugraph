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

import cupy

import pylibcugraph

from typing import Union, Optional, Dict, Tuple

from cugraph.utilities.utils import import_optional

from cugraph_dgl.typing import TensorType
from cugraph_dgl.utils.cugraph_conversion_utils import _cast_to_torch_tensor
from cugraph_dgl.features import WholeFeatureStore



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

        self.__sizes = {}
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
            if len(self.__edge_indices.keys(leaves_only=True, include_nested=True)) > 1:
                raise ValueError("Edge type is required for heterogeneous graphs.")
            return HOMOGENEOUS_EDGE_TYPE

        if isinstance(etype, Tuple[str, str, str]):
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
        number of nodes across workers. If the backing feature store is
        distributed (i.e. wholegraph), then only local features should
        be passed to the data argument.  If the backing feature store is
        replicated, then features for all nodes should be passed to the
        data argument, including those for nodes not on the local worker.

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
                        " match the global number of nodes but it does not match for "
                        f"{feature_name}."
                    )

        self.__num_nodes_dict[ntype] = global_num_nodes

        for feature_name, feature_tensor in data.items():
            self.__ndata_storage[ntype, feature_name] = self.__ndata_storage_type(
                feature_tensor, **self.__wg_kwargs
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
            if ids.max() + 1 > self.__num_nodes(ntype):
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
        )

        if data is not None:
            for attr_name, attr_tensor in data.items():
                self.__edata_storage[
                    dgl_can_edge_type, attr_name
                ] = self.__edata_storage_type(attr_tensor, **self.__wg_kwargs)

        num_edges = self.__edge_indices[dgl_can_edge_type].shape[1]
        if self.is_multi_gpu:
            num_edges = torch.tensor([num_edges], device='cuda', dtype=torch.int64)
            torch.distributed.all_reduce(num_edges, op=torch.distributed.ReduceOp.SUM)
        
        self.__num_edges_dict[dgl_can_edge_type] = int(num_edges)
        
        self.__graph = None
        self.__vertex_offsets = None

    def num_nodes(self, ntype: str = None) -> int:
        """
        Returns the number of nodes of ntype, or if ntype is not provided,
        the total number of nodes in the graph.
        """
        if ntype is None:
            return sum(self.__num_nodes_dict.values())

        return self.__num_nodes_dict[ntype]

    def number_of_nodes(self, ntype: str = None) -> int:
        """
        Alias for num_nodes.
        """
        return self.num_nodes(ntype=ntype)
    
    def num_edges(self, etype: Union[str, Tuple[str, str, str]]=None) -> int:
        """
        Returns the number of edges of etype, or if etype is not provided,
        the total number of edges in the graph.
        """
        if etype is None:
            return sum(self.__num_edges_dict.values())
    
        etype = self.to_canonical_etype(etype)
        return self.__num_edges_dict[etype]
    
    def number_of_edges(self, etype: Union[str, Tuple[str, str, str]]=None) -> int:
        """
        Alias for num_edges.
        """
        return self.num_edges(etype=etype)

    @property
    def is_homogeneous(self):
        return len(self.__num_edges_dict) <= 1 and len(self.__num_nodes_dict) <=1

    @property
    def _graph(self) -> Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph]:
        if self.__graph is None:
            edgelist_dict = self.__get_edgelist()

            if self.is_multi_gpu:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

                vertices_array = cupy.arange(
                    sum(self._num_vertices().values()), dtype="int64"
                )
                vertices_array = cupy.array_split(vertices_array, world_size)[rank]

                self.__graph = pylibcugraph.MGGraph(
                    self._resource_handle,
                    graph_properties,
                    [cupy.asarray(edgelist_dict["src"]).astype("int64")],
                    [cupy.asarray(edgelist_dict["dst"]).astype("int64")],
                    vertices_array=[vertices_array],
                    edge_id_array=[cupy.asarray(edgelist_dict["eid"])],
                    edge_type_array=[cupy.asarray(edgelist_dict["etp"])],
                )
            else:
                self.__graph = pylibcugraph.SGGraph(
                    self._resource_handle,
                    graph_properties,
                    cupy.asarray(edgelist_dict["src"]).astype("int64"),
                    cupy.asarray(edgelist_dict["dst"]).astype("int64"),
                    vertices_array=cupy.arange(
                        sum(self._num_vertices().values()), dtype="int64"
                    ),
                    edge_id_array=cupy.asarray(edgelist_dict["eid"]),
                    edge_type_array=cupy.asarray(edgelist_dict["etp"]),
                )

        return self.__graph