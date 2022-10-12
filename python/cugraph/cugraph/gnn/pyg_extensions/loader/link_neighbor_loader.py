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

from torch_geometric.loader.link_neighbor_loader import Dataset
from cugraph.gnn.pyg_extensions.loader.neighbor_loader import (
    EXPERIMENTAL__CuGraphNeighborSampler,
)
from cugraph.gnn.pyg_extensions.data.cugraph_store import EdgeLayout

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.typing import InputEdges, NumNeighbors, OptTensor

from torch_geometric.loader.utils import edge_type_to_str


class EXPERIMENTAL__CuGraphLinkNeighborSampler(EXPERIMENTAL__CuGraphNeighborSampler):
    def __init__(
        self,
        data,
        *args,
        neg_sampling_ratio: float = 0.0,
        num_src_nodes: Optional[int] = None,
        num_dst_nodes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(data, *args, **kwargs)
        self.neg_sampling_ratio = neg_sampling_ratio

        _, graph_store = data
        edge_attrs = graph_store.get_all_edge_attrs()
        edge_types = [attr.edge_type for attr in edge_attrs]

        # Edge label index is part of the graph.
        if self.input_type in edge_types:
            self.num_src_nodes, self.num_dst_nodes = edge_attrs[
                edge_types.index(self.input_type)
            ].size

        else:
            self.num_src_nodes = num_src_nodes
            self.num_dst_nodes = num_dst_nodes

    def _create_label(self, edge_label_index, edge_label):
        device = edge_label_index.device

        num_pos_edges = edge_label_index.size(1)
        num_neg_edges = int(num_pos_edges * self.neg_sampling_ratio)

        if num_neg_edges == 0:
            return edge_label_index, edge_label

        if edge_label is None:
            edge_label = torch.ones(num_pos_edges, device=device)
        else:
            assert edge_label.dtype == torch.long
            edge_label = edge_label + 1

        neg_row = torch.randint(self.num_src_nodes, (num_neg_edges,))
        neg_col = torch.randint(self.num_dst_nodes, (num_neg_edges,))
        neg_edge_label_index = torch.stack([neg_row, neg_col], dim=0)

        neg_edge_label = edge_label.new_zeros((num_neg_edges,) + edge_label.size()[1:])

        edge_label_index = torch.cat(
            [
                edge_label_index,
                neg_edge_label_index,
            ],
            dim=1,
        )

        edge_label = torch.cat([edge_label, neg_edge_label], dim=0)

        return edge_label_index, edge_label

    def __call__(self, query: List[Tuple[Tensor]]):
        query = [torch.tensor(s) for s in zip(*query)]
        if len(query) == 2:
            edge_label_index = torch.stack(query, dim=0)
            edge_label = None
        else:
            edge_label_index = torch.stack(query[:2], dim=0)
            edge_label = query[2]

        edge_label_index, edge_label = self._create_label(edge_label_index, edge_label)

        # CuGraph can pull vertices of any type
        # Edges can be from/to any arbitrary types (many to many)
        # Merge both source and destination node indices:
        query_nodes = edge_label_index.view(-1)
        query_nodes, reverse = query_nodes.unique(return_inverse=True)
        edge_label_index = reverse.view(2, -1)

        out = self.graph_store.neighbor_sample(
            query_nodes,
            self.num_neighbors,
            self.replace,
            self.directed,
            self.edge_types,
        )

        # Call cuGraph sampler
        return out + (edge_label_index, edge_label)


class EXPERIMENTAL__CuGraphLinkNeighborLoader(torch.utils.data.DataLoader):
    r"""A link-based data loader derived as an extension of the node-based
    :class:`torch_geometric.loader.NeighborLoader`.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, this loader first selects a sample of edges from the
    set of input edges :obj:`edge_label_index` (which may or not be edges in
    the original graph) and then constructs a subgraph from all the nodes
    present in this list by sampling :obj:`num_neighbors` neighbors in each
    iteration.

    Args:
        data (Tuple[FeatureStore, GraphStore]):
            The feature and graph stores for the cugraph graph.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        edge_label_index (Tensor or EdgeType or Tuple[EdgeType, Tensor]):
            The edge indices for which neighbors are sampled to create
            mini-batches.
            If set to :obj:`None`, all edges will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the edge type and corresponding edge indices.
            (default: :obj:`None`)
        edge_label (Tensor): The labels of edge indices for which neighbors are
            sampled. Must be the same length as the :obj:`edge_label_index`.
            If set to :obj:`None` then no labels are returned in the batch.
        num_src_nodes (optional, int): Not supported.
        num_dst_nodes (optional, int): Not support.
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            edges to the number of positive edges.
            If :obj:`edge_label` does not exist, it will be automatically
            created and represents a binary classification task
            (:obj:`1` = edge, :obj:`0` = no edge).
            If :obj:`edge_label` exists, it has to be a categorical label from
            :obj:`0` to :obj:`num_classes - 1`.
            After negative sampling, label :obj:`0` represents negative edges,
            and labels :obj:`1` to :obj:`num_classes` represent the labels of
            positive edges.
            Note that returned labels are of type :obj:`torch.float` for binary
            classification (to facilitate the ease-of-use of
            :meth:`F.binary_cross_entropy`) and of type
            :obj:`torch.long` for multi-class classification (to facilitate the
            ease-of-use of :meth:`F.cross_entropy`). (default: :obj:`0.0`).
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier timestamp than the center node. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): Not supported.
        filter_per_worker (bool, optional): If set to :obj:`True`, will filter
            the returning data in each worker's subprocess rather than in the
            main process.
            Setting this to :obj:`True` is generally not recommended:
            (1) it may result in too many open file handles,
            (2) it may slown down data loading,
            (3) it requires operating on CPU tensors.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """

    def __init__(
        self,
        data: Tuple[FeatureStore, GraphStore],
        num_neighbors: NumNeighbors,
        edge_label_index: InputEdges = None,
        edge_label: OptTensor = None,
        num_src_nodes: Optional[int] = None,
        num_dst_nodes: Optional[int] = None,
        replace: bool = False,
        directed: bool = True,
        neg_sampling_ratio: float = 0.0,
        time_attr: Optional[str] = None,
        transform: Callable = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        neighbor_sampler: Optional[EXPERIMENTAL__CuGraphLinkNeighborSampler] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if "dataset" in kwargs:
            del kwargs["dataset"]
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        if num_src_nodes is not None:
            raise ValueError("num_src_nodes parameter is not supported!")
        if num_dst_nodes is not None:
            raise ValueError("num_dst_nodes parameter is not supported!")
        if is_sorted is not False:
            raise ValueError("is_sorted parameter must be false!")

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.edge_label_index = edge_label_index
        self.edge_label = edge_label
        self.replace = replace
        self.directed = directed
        self.neg_sampling_ratio = neg_sampling_ratio
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neighbor_sampler = neighbor_sampler

        edge_type, edge_label_index = get_edge_label_index(data, edge_label_index)

        if neighbor_sampler is None:
            self.neighbor_sampler = EXPERIMENTAL__CuGraphLinkNeighborSampler(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=edge_type,
                neg_sampling_ratio=self.neg_sampling_ratio,
                time_attr=time_attr,
                share_memory=kwargs.get("num_workers", 0) > 0,
            )

        super().__init__(
            Dataset(edge_label_index, edge_label), collate_fn=self.collate_fn, **kwargs
        )

    def filter_fn(self, out: Any, add_empty_embeddings=True) -> Union[Data, HeteroData]:
        (
            node_dict,
            row_dict,
            col_dict,
            feature_dict,
            edge_label_index,
            edge_label,
        ) = out
        feature_store, graph_store = self.data

        # Construct a new `HeteroData` object:
        data = HeteroData()

        # Filter edge storage:
        # TODO support edge attributes
        for attr in graph_store.get_all_edge_attrs():
            key = edge_type_to_str(attr.edge_type)
            if key in row_dict and key in col_dict:
                edge_index = torch.stack([row_dict[key], col_dict[key]], dim=0)
                data[attr.edge_type].edge_index = edge_index

        # Filter node storage:
        for attr in feature_store.get_all_tensor_attrs():
            if attr.group_name in node_dict:
                attr.index = node_dict[attr.group_name]
                if attr.attr_name in feature_dict[attr.group_name]:
                    data[attr.group_name][attr.attr_name] = feature_dict[
                        attr.group_name
                    ][attr.attr_name]
                else:
                    data[attr.group_name][attr.attr_name] = torch.zeros_like(attr.index)

        edge_type = self.neighbor_sampler.input_type
        data[edge_type].edge_label_index = edge_label_index
        if edge_label is not None:
            data[edge_type].edge_label = edge_label

        return data if self.transform is None else self.transform(data)

    def collate_fn(self, index: Union[List[int], Tensor]) -> Any:
        out = self.neighbor_sampler(index)
        if self.filter_per_worker:
            # We execute `filter_fn` in the worker process.
            out = self.filter_fn(out)
        return out

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        # We execute `filter_fn` in the main process.
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def get_edge_label_index(
    data: Tuple[FeatureStore, GraphStore], edge_label_index: InputEdges
) -> Tuple[Optional[str], Tensor]:
    _, graph_store = data

    # Need the edge index in COO for LinkNeighborLoader:
    def _get_edge_index(edge_type):
        row, col = graph_store.get_edge_index(
            edge_type=edge_type, layout=EdgeLayout.COO, is_sorted=False
        )
        return torch.stack((row, col), dim=0)

    if isinstance(edge_label_index, str):
        if isinstance(edge_label_index[0], str):
            edge_type = edge_label_index
            return edge_type, _get_edge_index(edge_type)

    assert len(edge_label_index) == 2
    edge_type, edge_label_index = edge_label_index

    if edge_label_index is None:
        return edge_type, _get_edge_index(edge_type)

    return edge_type, edge_label_index
