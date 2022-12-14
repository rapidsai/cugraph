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

try:
    from cugraph_service.client.remote_graph_utils import import_optional, MissingModule
except ModuleNotFoundError:
    try:
        from cugraph.utilities.utils import import_optional, MissingModule
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "cuGraph extensions for PyG require cuGraph"
            "or cuGraph-Service to be installed."
        )

from cugraph_pyg.loader.dispatch import call_cugraph_algorithm

import cudf

dask_cudf = import_optional("dask_cudf")
torch_geometric = import_optional("torch_geometric")


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

    def sample_from_nodes(self, sampler_input):
        """
        Sample nodes using this CuGraphSampler's sampling method
        (which is set at initialization)
        and the input node data passed to this function.  Matches
        the interface provided by PyG's NodeSamplerInput.

        sampler_input: tuple(index, input_nodes, input_time)
            index: The sample indices to store as metadata
            input_nodes: Input nodes to pass to the sampler
            input_time: Node timestamps (if performing temporal
            sampling which is currently not supported)
        """
        index, input_nodes, input_time = sampler_input

        if input_time is not None:
            raise ValueError("Temporal sampling is currently unsupported in cuGraph")

        if self.__method == self.UNIFORM_NEIGHBOR:
            return self.__neighbor_sample(
                input_nodes, **self.__sampling_args, metadata=index
            )

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
        metadata=None,
        **kwargs,
    ):
        is_multi_gpu = self.__graph_store.is_multi_gpu
        if is_multi_gpu and isinstance(dask_cudf, MissingModule):
            raise ImportError("Cannot use a multi-GPU store without dask_cudf")
        if is_multi_gpu != self.__feature_store.is_multi_gpu:
            raise ValueError(
                f"Graph store multi-GPU is {is_multi_gpu}"
                f" but feature store multi-GPU is "
                f"{self.__feature_store.is_multi_gpu}"
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

        # FIXME resolve the directed/undirected issue
        G = self.__graph_store._subgraph([et[1] for et in edge_types])

        index = cudf.from_dlpack(index.__dlpack__())

        sampling_results = call_cugraph_algorithm(
            "uniform_neighbor_sample",
            G,
            index,
            # conversion required by cugraph api
            list(num_neighbors),
            replace,
        )

        concat_fn = dask_cudf.concat if is_multi_gpu else cudf.concat

        nodes_of_interest = concat_fn(
            [sampling_results.destinations, sampling_results.sources]
        ).unique()

        if is_multi_gpu:
            nodes_of_interest = nodes_of_interest.compute()

        # Get the grouped node index (for creating the renumbered grouped edge index)
        noi_index = self.__graph_store._get_vertex_groups_from_sample(nodes_of_interest)

        # Get the new edge index (by type as expected for HeteroData)
        # FIXME handle edge ids/types after the C++ updates
        row_dict, col_dict = self.__graph_store._get_renumbered_edge_groups_from_sample(
            sampling_results, noi_index
        )

        out = (noi_index, row_dict, col_dict, None)
        if isinstance(torch_geometric, MissingModule):
            return {"out": out, "metadata": metadata}
        else:
            return torch_geometric.sampler.base.HeteroSamplerOutput(
                *out, metadata=metadata
            )
