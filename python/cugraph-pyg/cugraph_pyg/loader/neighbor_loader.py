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

import warnings
import tempfile

from typing import Union, Tuple, Optional, Callable, List, Dict

import cugraph_pyg
from cugraph_pyg.loader import NodeLoader
from cugraph_pyg.sampler import BaseSampler

from cugraph.gnn import UniformNeighborSampler, DistSampleWriter
from cugraph.utilities.utils import import_optional

torch_geometric = import_optional("torch_geometric")


class NeighborLoader(NodeLoader):
    """
    Duck-typed version of torch_geometric.loader.NeighborLoader

    Node loader that implements the neighbor sampling
    algorithm used in GraphSAGE.
    """

    def __init__(
        self,
        data: Union[
            "torch_geometric.data.Data",
            "torch_geometric.data.HeteroData",
            Tuple[
                "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
            ],
        ],
        num_neighbors: Union[
            List[int], Dict["torch_geometric.typing.EdgeType", List[int]]
        ],
        input_nodes: "torch_geometric.typing.InputNodes" = None,
        input_time: "torch_geometric.typing.OptTensor" = None,
        replace: bool = False,
        subgraph_type: Union[
            "torch_geometric.typing.SubgraphType", str
        ] = "directional",
        disjoint: bool = False,
        temporal_strategy: str = "uniform",
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional["torch_geometric.sampler.NeighborSampler"] = None,
        directed: bool = True,  # Deprecated.
        batch_size: int = 16,
        directory: str = None,
        batches_per_partition=256,
        format: str = "parquet",
        compression: Optional[str] = None,
        local_seeds_per_call: Optional[int] = None,
        **kwargs,
    ):
        """
        data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
            See torch_geometric.loader.NeighborLoader.
        num_neighbors: List[int] or Dict[EdgeType, List[int]]
            Fanout values.
            See torch_geometric.loader.NeighborLoader.
        input_nodes: InputNodes
            Input nodes for sampling.
            See torch_geometric.loader.NeighborLoader.
        input_time: OptTensor (optional)
            See torch_geometric.loader.NeighborLoader.
        replace: bool (optional, default=False)
            Whether to sample with replacement.
            See torch_geometric.loader.NeighborLoader.
        subgraph_type: Union[SubgraphType, str] (optional, default='directional')
            The type of subgraph to return.
            Currently only 'directional' is supported.
            See torch_geometric.loader.NeighborLoader.
        disjoint: bool (optional, default=False)
            Whether to perform disjoint sampling.
            Currently unsupported.
            See torch_geometric.loader.NeighborLoader.
        temporal_strategy: str (optional, default='uniform')
            Currently only 'uniform' is suppported.
            See torch_geometric.loader.NeighborLoader.
        time_attr: str (optional, default=None)
            Used for temporal sampling.
            See torch_geometric.loader.NeighborLoader.
        weight_attr: str (optional, default=None)
            Used for biased sampling.
            See torch_geometric.loader.NeighborLoader.
        transform: Callable (optional, default=None)
            See torch_geometric.loader.NeighborLoader.
        transform_sampler_output: Callable (optional, default=None)
            See torch_geometric.loader.NeighborLoader.
        is_sorted: bool (optional, default=False)
            Ignored by cuGraph.
            See torch_geometric.loader.NeighborLoader.
        filter_per_worker: bool (optional, default=False)
            Currently ignored by cuGraph, but this may
            change once in-memory sampling is implemented.
            See torch_geometric.loader.NeighborLoader.
        neighbor_sampler: torch_geometric.sampler.NeighborSampler
            (optional, default=None)
            Not supported by cuGraph.
            See torch_geometric.loader.NeighborLoader.
        directed: bool (optional, default=True)
            Deprecated.
            See torch_geometric.loader.NeighborLoader.
        batch_size: int (optional, default=16)
            The number of input nodes per output minibatch.
            See torch.utils.dataloader.
        directory: str (optional, default=None)
            The directory where samples will be temporarily stored.
            It is recommend that this be set by the user, usually
            setting it to a tempfile.TemporaryDirectory with a context
            manager is a good option but depending on the filesystem,
            you may want to choose an alternative location with fast I/O
            intead.
            If not set, this will create a TemporaryDirectory that will
            persist until this object is garbage collected.
            See cugraph.gnn.DistSampleWriter.
        batches_per_partition: int (optional, default=256)
            The number of batches per partition if writing samples to
            disk.  Manually tuning this parameter is not recommended
            but reducing it may help conserve GPU memory.
            See cugraph.gnn.DistSampleWriter.
        format: str (optional, default='parquet')
            If writing samples to disk, they will be written in this
            file format.
            See cugraph.gnn.DistSampleWriter.
        compression: str (optional, default=None)
            The compression type to use if writing samples to disk.
            If not provided, it is automatically chosen.
        local_seeds_per_call: int (optional, default=None)
            The number of seeds to process within a single sampling call.
            Manually tuning this parameter is not recommended but reducing
            it may conserve GPU memory.  The total number of seeds processed
            per sampling call is equal to the sum of this parameter across
            all workers.  If not provided, it will be automatically
            calculated.
            See cugraph.gnn.DistSampler.
        **kwargs
            Other keyword arguments passed to the superclass.
        """

        subgraph_type = torch_geometric.sampler.base.SubgraphType(subgraph_type)

        if not directed:
            subgraph_type = torch_geometric.sampler.base.SubgraphType.induced
            warnings.warn(
                "The 'directed' argument is deprecated. "
                "Use subgraph_type='induced' instead."
            )
        if subgraph_type != torch_geometric.sampler.base.SubgraphType.directional:
            raise ValueError("Only directional subgraphs are currently supported")
        if disjoint:
            raise ValueError("Disjoint sampling is currently unsupported")
        if temporal_strategy != "uniform":
            warnings.warn("Only the uniform temporal strategy is currently supported")
        if neighbor_sampler is not None:
            raise ValueError("Passing a neighbor sampler is currently unsupported")
        if time_attr is not None:
            raise ValueError("Temporal sampling is currently unsupported")
        if weight_attr is not None:
            raise ValueError("Biased sampling is currently unsupported")
        if is_sorted:
            warnings.warn("The 'is_sorted' argument is ignored by cuGraph.")
        if not isinstance(data, (list, tuple)) or not isinstance(
            data[1], cugraph_pyg.data.GraphStore
        ):
            # Will eventually automatically convert these objects to cuGraph objects.
            raise NotImplementedError("Currently can't accept non-cugraph graphs")

        if directory is None:
            warnings.warn("Setting a directory to store samples is recommended.")
            self._tempdir = tempfile.TemporaryDirectory()
            directory = self._tempdir.name

        if compression is None:
            compression = "CSR"
        elif compression not in ["CSR", "COO"]:
            raise ValueError("Invalid value for compression (expected 'CSR' or 'COO')")

        writer = DistSampleWriter(
            directory=directory,
            batches_per_partition=batches_per_partition,
            format=format,
        )

        feature_store, graph_store = data
        sampler = BaseSampler(
            UniformNeighborSampler(
                graph_store._graph,
                writer,
                retain_original_seeds=True,
                fanout=num_neighbors,
                prior_sources_behavior="exclude",
                deduplicate_sources=True,
                compression=compression,
                compress_per_hop=False,
                with_replacement=replace,
                local_seeds_per_call=local_seeds_per_call,
            ),
            (feature_store, graph_store),
            batch_size=batch_size,
        )
        # TODO add heterogeneous support and pass graph_store._vertex_offsets

        super().__init__(
            (feature_store, graph_store),
            sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            batch_size=batch_size,
            **kwargs,
        )
