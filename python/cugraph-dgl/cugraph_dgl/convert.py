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
from __future__ import annotations

from cugraph_dgl import CuGraphStorage
from cugraph_dgl.utils.cugraph_conversion_utils import (
    add_edges_from_dgl_HeteroGraph,
    add_nodes_from_dgl_HeteroGraph,
)


def cugraph_storage_from_heterograph(g, single_gpu: bool = True) -> CuGraphStorage:
    """
    Convert DGL Graph to CuGraphStorage graph
    """
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    gs = CuGraphStorage(
        single_gpu=single_gpu, num_nodes_dict=num_nodes_dict, idtype=g.idtype
    )
    add_nodes_from_dgl_HeteroGraph(gs, g)
    add_edges_from_dgl_HeteroGraph(gs, g)
    return gs
