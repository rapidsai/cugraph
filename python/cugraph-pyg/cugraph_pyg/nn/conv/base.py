# Copyright (c) 2023, NVIDIA CORPORATION.
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
from typing import Optional, Tuple, Union

from cugraph.utilities.utils import import_optional

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")
ops_torch = import_optional("pylibcugraphops.pytorch")


class BaseConv(torch.nn.Module):  # pragma: no cover
    r"""An abstract base class for implementing cugraph-ops message passing layers."""

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    @staticmethod
    def to_csc(
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, int],
        Tuple[Tuple[torch.Tensor, torch.Tensor, int], torch.Tensor],
    ]:
        r"""Returns a CSC representation of an :obj:`edge_index` tensor to be
        used as input to cugraph-ops conv layers.

        Args:
            edge_index (torch.Tensor): The edge indices.
            size ((int, int), optional). The shape of :obj:`edge_index` in each
                dimension. (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features.
                (default: :obj:`None`)
        """
        if size is None:
            warnings.warn(
                f"Inferring the graph size from 'edge_index' causes "
                f"a decline in performance and does not work for "
                f"bipartite graphs. To suppress this warning, pass "
                f"the 'size' explicitly in '{__name__}.to_csc()'."
            )
            num_src_nodes = num_dst_nodes = int(edge_index.max()) + 1
        else:
            num_src_nodes, num_dst_nodes = size

        row, col = edge_index
        col, perm = torch_geometric.utils.index_sort(col, max_value=num_dst_nodes)
        row = row[perm]

        colptr = torch_geometric.utils.sparse.index2ptr(col, num_dst_nodes)

        if edge_attr is not None:
            return (row, colptr, num_src_nodes), edge_attr[perm]

        return row, colptr, num_src_nodes

    def get_cugraph(
        self,
        csc: Tuple[torch.Tensor, torch.Tensor, int],
        bipartite: bool = False,
        max_num_neighbors: Optional[int] = None,
    ) -> ops_torch.CSC:
        r"""Constructs a :obj:`cugraph-ops` graph object from CSC representation.
        Supports both bipartite and non-bipartite graphs.

        Args:
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`to_csc` method to convert an :obj:`edge_index`
                representation to the desired format.
            bipartite (bool): If set to :obj:`True`, will create the bipartite
                structure in cugraph-ops. (default: :obj:`False`)
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, will be computed on-the-fly,
                leading to slightly worse performance. (default: :obj:`None`)
        """
        row, colptr, num_src_nodes = csc

        if not row.is_cuda:
            raise RuntimeError(
                f"'{self.__class__.__name__}' requires GPU-"
                f"based processing (got CPU tensor)"
            )

        if max_num_neighbors is None:
            max_num_neighbors = -1

        return ops_torch.CSC(
            offsets=colptr,
            indices=row,
            num_src_nodes=num_src_nodes,
            dst_max_in_degree=max_num_neighbors,
            is_bipartite=bipartite,
        )

    def get_typed_cugraph(
        self,
        csc: Tuple[torch.Tensor, torch.Tensor, int],
        edge_type: torch.Tensor,
        num_edge_types: Optional[int] = None,
        bipartite: bool = False,
        max_num_neighbors: Optional[int] = None,
    ) -> ops_torch.HeteroCSC:
        r"""Constructs a typed :obj:`cugraph` graph object from a CSC
        representation where each edge corresponds to a given edge type.
        Supports both bipartite and non-bipartite graphs.

        Args:
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`to_csc` method to convert an :obj:`edge_index`
                representation to the desired format.
            edge_type (torch.Tensor): The edge type.
            num_edge_types (int, optional): The maximum number of edge types.
                When not given, will be computed on-the-fly, leading to
                slightly worse performance. (default: :obj:`None`)
            bipartite (bool): If set to :obj:`True`, will create the bipartite
                structure in cugraph-ops. (default: :obj:`False`)
            max_num_neighbors (int, optional): The maximum number of neighbors
                of a target node. It is only effective when operating in a
                bipartite graph. When not given, will be computed on-the-fly,
                leading to slightly worse performance. (default: :obj:`None`)
        """
        if num_edge_types is None:
            num_edge_types = int(edge_type.max()) + 1

        if max_num_neighbors is None:
            max_num_neighbors = -1

        row, colptr, num_src_nodes = csc
        edge_type = edge_type.int()

        return ops_torch.HeteroCSC(
            offsets=colptr,
            indices=row,
            edge_types=edge_type,
            num_src_nodes=num_src_nodes,
            num_edge_types=num_edge_types,
            dst_max_in_degree=max_num_neighbors,
            is_bipartite=bipartite,
        )

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        csc: Tuple[torch.Tensor, torch.Tensor, int],
    ) -> torch.Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The node features.
            csc ((torch.Tensor, torch.Tensor, int)): A tuple containing the CSC
                representation of a graph, given as a tuple of
                :obj:`(row, colptr, num_src_nodes)`. Use the
                :meth:`to_csc` method to convert an :obj:`edge_index`
                representation to the desired format.
        """
        raise NotImplementedError
