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

import cugraph


from typing import Tuple, List, Union, Sequence, Dict

from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.data.cugraph_store import TensorType

from cugraph.utilities.utils import import_optional, MissingModule
import cudf

dask_cudf = import_optional("dask_cudf")
torch_geometric = import_optional("torch_geometric")

torch = import_optional("torch")

HeteroSamplerOutput = (
    None
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.sampler.base.HeteroSamplerOutput
)


def _sampler_output_from_sampling_results(
    sampling_results: cudf.DataFrame,
    graph_store: CuGraphStore,
    metadata: Sequence = None,
) -> Union[HeteroSamplerOutput, Dict[str, dict]]:
    """
    Parameters
    ----------
    sampling_results: cudf.DataFrame
        The dataframe containing sampling results.
    graph_store: CuGraphStore
        The graph store containing the structure of the sampled graph.
    metadata: Tensor
        The metadata for the sampled batch.

    Returns
    -------
    HeteroSamplerOutput, if PyG is installed.
    dict, if PyG is not installed.
    """
    nodes_of_interest = cudf.concat(
        [sampling_results.destinations, sampling_results.sources]
    ).unique()

    # Get the grouped node index (for creating the renumbered grouped edge index)
    noi_index = graph_store._get_vertex_groups_from_sample(nodes_of_interest)

    # Get the new edge index (by type as expected for HeteroData)
    # FIXME handle edge ids/types after the C++ updates
    row_dict, col_dict = graph_store._get_renumbered_edge_groups_from_sample(
        sampling_results, noi_index
    )

    out = (noi_index, row_dict, col_dict, None)

    # FIXME no longer allow torch_geometric to be missing.
    if isinstance(torch_geometric, MissingModule):
        return {"out": out, "metadata": metadata}
    else:
        return HeteroSamplerOutput(*out, metadata=metadata)


class EXPERIMENTAL__CuGraphSampler:
    """
    Duck-typed version of PyG's BaseSampler
    """

    UNIFORM_NEIGHBOR = "uniform_neighbor"
    SAMPLING_METHODS = [
        UNIFORM_NEIGHBOR,
    ]

    def __init__(
        self,
        data: Tuple[CuGraphStore, CuGraphStore],
        method: str = UNIFORM_NEIGHBOR,
        **kwargs,
    ):
        if method not in self.SAMPLING_METHODS:
            raise ValueError(f"{method} is not a valid sampling method")
        self.__method = method
        self.__sampling_args = kwargs

        fs, gs = data
        self.__feature_store = fs
        self.__graph_store = gs

    # FIXME Make HeteroSamplerOutput the only return type
    # after PyG becomes a hard requirement
    def sample_from_nodes(
        self, sampler_input: Tuple[TensorType, TensorType, TensorType]
    ) -> Union[HeteroSamplerOutput, dict]:
        """
        Sample nodes using this CuGraphSampler's sampling method
        (which is set at initialization)
        and the input node data passed to this function.  Matches
        the interface provided by PyG's NodeSamplerInput.

        Parameters
        ----------
        sampler_input: tuple(index, input_nodes, input_time)
            index: The sample indices to store as metadata
            input_nodes: Input nodes to pass to the sampler
            input_time: Node timestamps (if performing temporal
            sampling which is currently not supported)

        Returns
        -------
        HeteroSamplerOutput, if PyG is installed.
        dict, if PyG is not installed.
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
    def method(self) -> str:
        return self.__method

    @property
    def edge_permutation(self):
        return None

    """
    SAMPLER IMPLEMENTATIONS
    """

    def __neighbor_sample(
        self,
        index: TensorType,
        num_neighbors: List[int],
        replace: bool = True,
        directed: bool = True,
        edge_types: List[str] = None,
        metadata=None,
        **kwargs,
    ) -> Union[dict, HeteroSamplerOutput]:
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

        if backend == "torch" and not index.is_cuda:
            index = index.cuda()

        G = self.__graph_store._subgraph(edge_types)

        index = cudf.Series(index)

        sample_fn = (
            cugraph.dask.uniform_neighbor_sample
            if self.__graph_store._is_delayed
            else cugraph.uniform_neighbor_sample
        )

        sampling_results = sample_fn(
            G,
            index,
            # conversion required by cugraph api
            list(num_neighbors),
            replace,
            with_edge_properties=True,
        )

        if self.__graph_store._is_delayed:
            sampling_results = sampling_results.compute()

        return _sampler_output_from_sampling_results(
            sampling_results, self.__graph_store, metadata
        )
