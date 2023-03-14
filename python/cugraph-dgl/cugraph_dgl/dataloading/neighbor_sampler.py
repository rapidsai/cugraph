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
from typing import Sequence


class NeighborSampler:
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.
    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.
    Parameters
    ----------
    fanouts_per_layer : int
        List of neighbors to sample for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
        If -1 is provided then all inbound/outbound edges
        of that edge type will be included.
    edge_dir : str, default ``'in'``
        Can be either ``'in' `` where the neighbors will be sampled according to
        incoming edges, or ``'out'`` for outgoing edges
    replace : bool, default False
        Whether to sample with replacement
    Examples
    --------
    **Node classification**
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):
    >>> sampler = cugraph_dgl.dataloading.NeighborSampler([5, 10, 15])
    >>> dataloader = cugraph_dgl.dataloading.DataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)
    """

    def __init__(
        self,
        fanouts_per_layer: Sequence[int],
        edge_dir: str = "in",
        replace: bool = False,
    ):
        self.fanouts = fanouts_per_layer
        reverse_fanouts = fanouts_per_layer.copy()
        reverse_fanouts.reverse()
        self._reversed_fanout_vals = reverse_fanouts

        self.edge_dir = edge_dir
        self.replace = replace
