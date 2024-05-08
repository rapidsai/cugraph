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

import numpy as np
import cupy
import cudf
import pandas

import pylibcugraph

from cugraph.utilities.utils import import_optional, MissingModule
from cugraph.gnn.comms import cugraph_comms_get_raft_handle

from typing import Union, Optional, List, Dict


# Have to use import_optional even though these are required
# dependencies in order to build properly.
torch_geometric = import_optional("torch_geometric")
torch = import_optional("torch")
tensordict = import_optional("tensordict")

TensorType = Union["torch.Tensor", cupy.ndarray, np.ndarray, cudf.Series, pandas.Series]


class GraphStore(
    object
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.data.GraphStore
):
    """
    This object uses lazy graph creation.  Users can repeatedly call
    put_edge_index, and the tensors won't be converted into a cuGraph
    graph until one is needed (i.e. when creating a loader).
    """

    def __init__(self, is_multi_gpu: bool = False):
        self.__edge_indices = tensordict.TensorDict({}, batch_size=(2,))
        self.__sizes = {}
        self.__graph = None
        self.__vertex_offsets = None
        self.__handle = None
        self.__is_multi_gpu = is_multi_gpu

        super().__init__()

    def _put_edge_index(
        self,
        edge_index: "torch_geometric.typing.EdgeTensorType",
        edge_attr: "torch_geometric.data.EdgeAttr",
    ) -> bool:
        if edge_attr.layout != torch_geometric.data.graph_store.EdgeLayout.COO:
            raise ValueError("Only COO format supported")

        if isinstance(edge_index, (cupy.ndarray, cudf.Series)):
            edge_index = torch.as_tensor(edge_index, device="cuda")
        elif isinstance(edge_index, (np.ndarray)):
            edge_index = torch.as_tensor(edge_index, device="cpu")
        elif isinstance(edge_index, pandas.Series):
            edge_index = torch.as_tensor(edge_index.values, device="cpu")
        elif isinstance(edge_index, cudf.Series):
            edge_index = torch.as_tensor(edge_index.values, device="cuda")

        self.__edge_indices[edge_attr.edge_type] = torch.stack(
            [edge_index[0], edge_index[1]]
        )
        self.__sizes[edge_attr.edge_type] = edge_attr.size

        # invalidate the graph
        self.__graph = None
        self.__vertex_offsets = None
        return True

    def _get_edge_index(
        self, edge_attr: "torch_geometric.data.EdgeAttr"
    ) -> Optional["torch_geometric.typing.EdgeTensorType"]:
        ei = torch_geometric.EdgeIndex(self.__edge_indices[edge_attr.edge_type])

        if edge_attr.layout == "csr":
            return ei.sort_by("row").values.get_csr()
        elif edge_attr.layout == "csc":
            return ei.sort_by("col").values.get_csc()

        return ei

    def _remove_edge_index(self, edge_attr: "torch_geometric.data.EdgeAttr") -> bool:
        del self.__edge_indices[edge_attr.edge_type]

        # invalidate the graph
        self.__graph = None
        return True

    def get_all_edge_attrs(self) -> List["torch_geometric.data.EdgeAttr"]:
        attrs = []
        for et in self.__edge_indices.keys(leaves_only=True, include_nested=True):
            attrs.append(
                torch_geometric.data.EdgeAttr(
                    edge_type=et, layout="coo", is_sorted=False, size=self.__sizes[et]
                )
            )

        return attrs

    @property
    def is_multi_gpu(self):
        return self.__is_multi_gpu

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

    @property
    def _graph(self) -> Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph]:
        graph_properties = pylibcugraph.GraphProperties(
            is_multigraph=True, is_symmetric=False
        )

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
                    [cupy.asarray(edgelist_dict["src"])],
                    [cupy.asarray(edgelist_dict["dst"])],
                    vertices_array=[vertices_array],
                    edge_id_array=[cupy.asarray(edgelist_dict["eid"])],
                    edge_type_array=[cupy.asarray(edgelist_dict["etp"])],
                )
            else:
                self.__graph = pylibcugraph.SGGraph(
                    self._resource_handle,
                    graph_properties,
                    cupy.asarray(edgelist_dict["src"]),
                    cupy.asarray(edgelist_dict["dst"]),
                    vertices_array=cupy.arange(
                        sum(self._num_vertices.values()), dtype="int64"
                    ),
                    edge_id_array=cupy.asarray(edgelist_dict["eid"]),
                    edge_type_array=cupy.asarray(edgelist_dict["etp"]),
                )

        return self.__graph

    def _num_vertices(self) -> Dict[str, int]:
        num_vertices = {}
        for edge_attr in self.get_all_edge_attrs():
            if edge_attr.size is not None:
                num_vertices[edge_attr.edge_type[0]] = (
                    max(num_vertices[edge_attr.edge_type[0]], edge_attr.size[0])
                    if edge_attr.edge_type[0] in num_vertices
                    else edge_attr.size[0]
                )
                num_vertices[edge_attr.edge_type[2]] = (
                    max(num_vertices[edge_attr.edge_type[2]], edge_attr.size[1])
                    if edge_attr.edge_type[2] in num_vertices
                    else edge_attr.size[1]
                )
            else:
                if edge_attr.edge_type[0] not in num_vertices:
                    num_vertices[edge_attr.edge_type[0]] = (
                        self.__edge_indices[edge_attr.edge_type][0].max() + 1
                    )
                if edge_attr.edge_type[2] not in num_vertices:
                    num_vertices[edge_attr.edge_type[1]] = (
                        self.__edge_indices[edge_attr.edge_type][1].max() + 1
                    )

        return num_vertices

    @property
    def _vertex_offsets(self) -> Dict[str, int]:
        if self.__vertex_offsets is None:
            num_vertices = self._num_vertices()
            ordered_keys = sorted(list(num_vertices.keys()))
            self.__vertex_offsets = {}
            offset = 0
            for vtype in ordered_keys:
                self.__vertex_offsets[vtype] = offset
                offset += num_vertices[vtype]

        return dict(self.__vertex_offsets)

    @property
    def is_homogeneous(self) -> bool:
        return len(self._vertex_offsets) == 1

    def __get_edgelist(self):
        """
        Returns
        -------
        Dict[str, torch.Tensor] with the following keys:
            src: source vertices (int64)
                Note that src is the 2nd element of the PyG edge index.
            dst: destination vertices (int64)
                Note that dst is the 1st element of the PyG edge index.
            eid: edge ids for each edge (int64)
                Note that these start from 0 for each edge type.
            etp: edge types for each edge (int32)
                Note that these are in lexicographic order.
        """
        sorted_keys = sorted(
            list(self.__edge_indices.keys(leaves_only=True, include_nested=True))
        )

        # note that this still follows the PyG convention of (dst, rel, src)
        # i.e. (author, writes, paper): [[0,1,2],[2,0,1]] is referring to a
        # cuGraph graph where (paper 2) -> (author 0), (paper 0) -> (author 1),
        # and (paper 1) -> (author 0)
        edge_index = torch.concat(
            [
                torch.stack(
                    [
                        self.__edge_indices[dst_type, rel_type, src_type][0]
                        + self._vertex_offsets[dst_type],
                        self.__edge_indices[dst_type, rel_type, src_type][1]
                        + self._vertex_offsets[src_type],
                    ]
                )
                for (dst_type, rel_type, src_type) in sorted_keys
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

        if self.is_multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            num_edges_t = torch.tensor(
                [self.__edge_indices[et].shape[1] for et in sorted_keys], device="cuda"
            )
            num_edges_all_t = torch.empty(
                world_size, num_edges_t.numel(), dtype=torch.int64, device="cuda"
            )
            torch.distributed.all_gather_into_tensor(num_edges_all_t, num_edges_t)

            if rank > 0:
                start_offsets = num_edges_all_t[:rank].T.sum(axis=1)
                edge_id_array = torch.concat(
                    [
                        torch.arange(
                            start_offsets[i],
                            start_offsets[i] + num_edges_all_t[rank][i],
                            dtype=torch.int64,
                            device="cuda",
                        )
                        for i in range(len(sorted_keys))
                    ]
                )
            else:
                edge_id_array = torch.concat(
                    [
                        torch.arange(
                            self.__edge_indices[et].shape[1],
                            dtype=torch.int64,
                            device="cuda",
                        )
                        for et in sorted_keys
                    ]
                )

        else:
            # single GPU
            edge_id_array = torch.concat(
                [
                    torch.arange(
                        self.__edge_indices[et].shape[1],
                        dtype=torch.int64,
                        device="cuda",
                    )
                    for et in sorted_keys
                ]
            )

        return {
            "dst": edge_index[0],
            "src": edge_index[1],
            "etp": edge_type_array,
            "eid": edge_id_array,
        }
