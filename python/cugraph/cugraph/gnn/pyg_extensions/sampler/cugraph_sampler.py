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

from cugraph.utilities.utils import MissingModule, import_optional
from cugraph.gnn.pyg_extensions.loader.dispatch import call_cugraph_algorithm

import cudf
import cupy

dask_cudf = import_optional("dask_cudf")


class EXPERIMENTAL__CuGraphSampler:
    """
    Duck-typed version of PyG's BaseSampler
    """

    UNIFORM_NEIGHBOR = "uniform_neighbor"
    SAMPLING_METHODS = [
        UNIFORM_NEIGHBOR,
    ]

    def __init__(self, data, method=UNIFORM_NEIGHBOR, **kwargs):
        if method not in self.SAMPLING_METHODS:
            raise ValueError(f"{method} is not a valid sampling method")
        self.__method = method
        self.__sampling_args = kwargs

        fs, gs = data
        self.__feature_store = fs
        self.__graph_store = gs

    def sample_from_nodes(self, index):
        """
        index: input node tensor
        """
        if self.__method == self.UNIFORM_NEIGHBOR:
            return self.__neighbor_sample(index, **self.__sampling_args)

    def sample_from_edges(self, index):
        raise NotImplementedError("Edge sampling currently unsupported")

    @property
    def method(self):
        return self.__method

    @property
    def edge_permutation(self):
        return None

    """
    SAMPLER IMPLEMENTATIONS
    """

    def __neighbor_sample(
        self,
        index,
        num_neighbors,
        replace=True,
        directed=True,
        edge_types=None,
        **kwargs,
    ):
        is_mg = self.__graph_store.is_mg
        if is_mg and dask_cudf == MissingModule:
            raise ImportError("Cannot use a multi-GPU store without dask_cudf")
        if is_mg != self.__feature_store.is_mg:
            raise ValueError(
                f"Graph store multi-GPU is {is_mg}"
                f" but feature store multi-GPU is "
                f"{self.__feature_store.is_mg}"
            )

        backend = self.__graph_store.backend
        if backend != self.__feature_store.backend:
            raise ValueError(
                f"Graph store backend {backend}"
                f"does not match feature store "
                f"backend {self.__feature_store.backend}"
            )

        if not directed:
            raise ValueError("Undirected sampling not currently supported")

        if edge_types is None:
            edge_types = [
                attr.edge_type for attr in self.__graph_store.get_all_edge_attrs()
            ]

        if isinstance(num_neighbors, dict):
            # FIXME support variable num neighbors per edge type
            num_neighbors = list(num_neighbors.values())[0]

        # FIXME eventually get uniform neighbor sample to accept longs
        if backend == "torch" and not index.is_cuda:
            index = index.cuda()
        index = cupy.from_dlpack(index.__dlpack__())

        # FIXME resolve the directed/undirected issue
        G = self.__graph_store._subgraph([et[1] for et in edge_types])

        index = cudf.Series(index)

        sampling_results = call_cugraph_algorithm(
            "uniform_neighbor_sample",
            G,
            index,
            # conversion required by cugraph api
            list(num_neighbors),
            replace,
        )

        concat_fn = dask_cudf.concat if is_mg else cudf.concat

        nodes_of_interest = concat_fn(
            [sampling_results.destinations, sampling_results.sources]
        ).unique()

        if is_mg:
            nodes_of_interest = nodes_of_interest.compute()

        # Get the node index (for creating the edge index),
        # the node type groupings, and the node properties.
        (
            noi_index,
            noi_groups,
            noi_tensors,
        ) = self.__feature_store._get_renumbered_vertex_data_from_sample(
            nodes_of_interest
        )

        # Get the new edge index (by type as expected for HeteroData)
        # FIXME handle edge ids
        row_dict, col_dict = self.__graph_store._get_renumbered_edges_from_sample(
            sampling_results, noi_index
        )

        return (noi_groups, row_dict, col_dict, noi_tensors)
