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

from cugraph_dgl.typing import TensorType
from cugraph_dgl.utils.cugraph_conversion_utils import _cast_to_torch_tensor

from typing import Union, Optional, Dict, Tuple

# Have to use import_optional even though these are required
# dependencies in order to build properly.
dgl = import_optional("dgl")
torch = import_optional("torch")
tensordict = import_optional("tensordict")

HOMOGENEOUS_EDGE_TYPE = ('n','e','n')

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

    def __init__(self, is_multi_gpu: bool=False):
        """
        Parameters
        ----------
        is_multi_gpu: bool (optional, default=False)
            Specifies whether this graph is distributed across GPUs.
        """

        self.__edge_indices = tensordict.TensorDict({}, batch_size=(2,))
        self.__sizes = {}
        self.__graph = None
        self.__vertex_offsets = None
        self.__handle = None
        self.__is_multi_gpu = is_multi_gpu

    def to_canonical_etype(self, etype: Union[str, Tuple[str, str, str]]) -> Tuple[str, str, str]:
        if etype is None:
            if len(self.__edge_indices.keys(leaves_only=True,include_nested=True)) > 1:
                raise ValueError("Edge type is required for heterogeneous graphs.")
            return HOMOGENEOUS_EDGE_TYPE

        if isinstance(etype, Tuple[str, str, str]):
            return etype
    
        for src_type, rel_type, dst_type in self.__edge_indices.keys(leaves_only=True,include_nested=True):
            if etype == rel_type:
                return (src_type, rel_type, dst_type)
    
        raise ValueError(
            "Unknown relation type " + etype
        )
    
    def add_edges(self, u: TensorType, v: TensorType, data:Optional[Dict[str, TensorType]]=None, etype:Optional[Union[str, Tuple[str, str, str]]]=None) -> None:
        """
        Adds edges to this graph.

        Parameters
        ----------
        u: TensorType
            1d tensor of source vertex ids.
        v: TensorType
            1d tensor of destination vertex ids.
        data: Dict[str, TensorType] (optional, default=None)
            Dictionary containing edge features for the new edges.
        etype: Union[str, Tuple[str, str, str]]
            The edge type of the edges being inserted.  Not required
            for homogeneous graphs, which have only one edge type.
        """

        dgl_can_edge_type = self.to_canonical_etype(etype)

        new_edges = torch.stack([
            _cast_to_torch_tensor(u),
            _cast_to_torch_tensor(v),
        ])

        if dgl_can_edge_type in self.__edge_indices.keys(leaves_only=True, include_nested=True):
            self.__edge_indices[dgl_can_edge_type] = torch.concat([
                self.__edge_indices[dgl_can_edge_type],
                new_edges,
            ], dim=1)
        else:
            self.__edge_indices[dgl_can_edge_type] = new_edges
        
        if data is not None:
            