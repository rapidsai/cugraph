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

from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.neighbor_loader import get_input_nodes
from torch_geometric.loader.utils import (
    edge_type_to_str,
    filter_custom_store
)
from torch_geometric.typing import InputNodes, NumNeighbors


class EXPERIMENTAL__CuGraphNeighborSampler:
    def __init__(
        self,
        data: Tuple[FeatureStore, GraphStore],
        num_neighbors: NumNeighbors,
        replace: bool = False,
        directed: bool = True,
        input_type: Optional[Any] = None,
        time_attr: Optional[str] = None,
        is_sorted: bool = False,
        share_memory: bool = False,
    ):
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.directed = directed
        self.node_time = None

        feature_store, graph_store = data

        self.node_time_dict = None
        if time_attr is not None:
            # We need to obtain all features with 'attr_name=time_attr'
            # from the feature store and store them in node_time_dict. To
            # do so, we make an explicit feature store GET call here with
            # the relevant 'TensorAttr's
            time_attrs = [
                attr for attr in feature_store.get_all_tensor_attrs()
                if attr.attr_name == time_attr
            ]
            for attr in time_attrs:
                attr.index = None
            time_tensors = feature_store.multi_get_tensor(time_attrs)
            self.node_time_dict = {
                time_attr.group_name: time_tensor
                for time_attr, time_tensor in zip(time_attrs, time_tensors)
            }

        # Obtain all node and edge metadata:
        node_attrs = feature_store.get_all_tensor_attrs()
        edge_attrs = graph_store.get_all_edge_attrs()

        self.node_types = list(
            set(node_attr.group_name for node_attr in node_attrs))
        self.edge_types = list(
            set(edge_attr.edge_type for edge_attr in edge_attrs))

        # Set other required parameters:
        self._set_num_neighbors_and_num_hops(num_neighbors)

        assert input_type is not None
        self.input_type = input_type

        self.feature_store = feature_store
        self.graph_store = graph_store

    def _set_num_neighbors_and_num_hops(self, num_neighbors):
        if isinstance(num_neighbors, (list, tuple)):
            num_neighbors = {key: num_neighbors for key in self.edge_types}
        assert isinstance(num_neighbors, dict)
        self.num_neighbors = {
            edge_type_to_str(key): value
            for key, value in num_neighbors.items()
        }
        # Add at least one element to the list to ensure `max` is well-defined
        self.num_hops = max([0] + [len(v) for v in num_neighbors.values()])

    def __call__(self, index: Union[List[int], Tensor]):
        if not isinstance(index, torch.LongTensor):
            index = torch.LongTensor(index)

        out = self.graph_store.neighbor_sample(
            index,
            self.num_neighbors,
            self.replace,
            self.directed,
            self.edge_types
        )

        # call cugraph sampler
        return out + (index.numel(), )


class EXPERIMENTAL__CuGraphNeighborLoader(torch.utils.data.DataLoader):
    r"""A data loader that performs neighbor sampling as introduced in the
    `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`num_neighbors` denotes how much neighbors are
    sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    By default, the data loader will only include the edges that were
    originally sampled (:obj:`directed = True`).
    This option should only be used in case the number of hops is equivalent to
    the number of GNN layers.
    In case the number of GNN layers is greater than the number of hops,
    consider setting :obj:`directed = False`, which will include all edges
    between all sampled nodes (but is slightly slower as a result).

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.NeighborLoader`, see
        `examples/hetero/to_hetero_mag.py <https://github.com/pyg-team/
        pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py>`_.

    The :class:`~torch_geometric.loader.NeighborLoader` will return subgraphs
    where global node indices are mapped to local indices corresponding to this
    specific subgraph. However, often times it is desired to map the nodes of
    the current subgraph back to the global node indices. A simple trick to
    achieve this is to include this mapping as part of the :obj:`data` object:

    .. code-block:: python

        # Assign each node its global node index:
        data.n_id = torch.arange(data.num_nodes)

        loader = NeighborLoader(data, ...)
        sampled_data = next(iter(loader))
        print(sampled_data.n_id)

    Args:
        data (torch_geometric.data.Data or torch_geometric.data.HeteroData):
            The :class:`~torch_geometric.data.Data` or
            :class:`~torch_geometric.data.HeteroData` graph object.
        num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]): The
            number of neighbors to sample for each node in each iteration.
            In heterogeneous graphs, may also take in a dictionary denoting
            the amount of neighbors to sample for each individual edge type.
            If an entry is set to :obj:`-1`, all neighbors will be included.
        input_nodes (torch.Tensor or str or Tuple[str, torch.Tensor]): The
            indices of nodes for which neighbors are sampled to create
            mini-batches.
            Needs to be either given as a :obj:`torch.LongTensor` or
            :obj:`torch.BoolTensor`.
            If set to :obj:`None`, all nodes will be considered.
            In heterogeneous graphs, needs to be passed as a tuple that holds
            the node type and node indices. (default: :obj:`None`)
        replace (bool, optional): If set to :obj:`True`, will sample with
            replacement. (default: :obj:`False`)
        directed (bool, optional): If set to :obj:`False`, will include all
            edges between all sampled nodes. (default: :obj:`True`)
        time_attr (str, optional): The name of the attribute that denotes
            timestamps for the nodes in the graph.
            If set, temporal sampling will be used such that neighbors are
            guaranteed to fulfill temporal constraints, *i.e.* neighbors have
            an earlier timestamp than the center node. (default: :obj:`None`)
        transform (Callable, optional): A function/transform that takes in
            a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        is_sorted (bool, optional): Not supported
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
        input_nodes: InputNodes = None,
        replace: bool = False,
        directed: bool = True,
        time_attr: Optional[str] = None,
        transform: Callable = None,
        is_sorted: bool = False,
        filter_per_worker: bool = False,
        neighbor_sampler: Optional[
            EXPERIMENTAL__CuGraphNeighborSampler
        ] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        if 'dataset' in kwargs:
            del kwargs['dataset']
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        if is_sorted is not False:
            raise ValueError('is_sorted must be false')

        self.data = data

        # Save for PyTorch Lightning < 1.6:
        self.num_neighbors = num_neighbors
        self.input_nodes = input_nodes
        self.replace = replace
        self.directed = directed
        self.transform = transform
        self.filter_per_worker = filter_per_worker
        self.neighbor_sampler = neighbor_sampler

        node_type, input_nodes = get_input_nodes(data, input_nodes)

        if neighbor_sampler is None:
            self.neighbor_sampler = EXPERIMENTAL__CuGraphNeighborSampler(
                data,
                num_neighbors,
                replace,
                directed,
                input_type=node_type,
                time_attr=time_attr,
                share_memory=kwargs.get('num_workers', 0) > 0,
            )

        super().__init__(input_nodes, collate_fn=self.collate_fn, **kwargs)

    def filter_fn(self, out: Any) -> HeteroData:
        node_dict, row_dict, col_dict, edge_dict, batch_size = out
        feature_store, graph_store = self.data
        data = filter_custom_store(feature_store, graph_store, node_dict,
                                   row_dict, col_dict, edge_dict)
        data[self.neighbor_sampler.input_type].batch_size = batch_size

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
        return f'{self.__class__.__name__}()'
