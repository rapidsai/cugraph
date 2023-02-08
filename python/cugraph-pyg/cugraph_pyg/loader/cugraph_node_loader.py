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

import tempfile

import os

import cupy
import cudf

from cugraph.experimental.gnn import BulkSampler
from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.loader.filter import _filter_cugraph_store
from cugraph_pyg.sampler.cugraph_sampler import _sampler_output_from_sampling_results

from typing import Union, Tuple, Sequence, List

torch_geometric = import_optional("torch_geometric")


class EXPERIMENTAL__BulkSampleLoader:
    def __init__(
        self,
        feature_store: CuGraphStore,
        graph_store: CuGraphStore,
        all_indices: Union[Sequence, int],
        batch_size: int = 0,
        shuffle=False,
        edge_types: Sequence[Tuple[str]] = None,
        directory=None,
        rank=0,
        starting_batch_id=0,
        batches_per_partition=100,
        # Sampler args
        num_neighbors: List[int] = [1, 1],
        replace: bool = True,
        # Other kwargs for the BulkSampler
        **kwargs,
    ):
        """
        Executes a bulk sampling job immediately upon creation.
        Allows iteration over the returned results.

        Parameters
        ----------
        feature_store: CuGraphStore
            The feature store containing features for the graph.

        graph_store: CuGraphStore
            The graph store containing the graph structure.

        all_indices: Union[Tensor, int]
            The input nodes associated with this sampler.
            If this is an integer N , this loader will load N batches
            from disk rather than performing sampling in memory.

        batch_size: int
            The number of input nodes per sampling batch.
            Generally required unless loading already-sampled
            data from disk.

        shuffle: bool (optional, default=False)
            Whether to shuffle the input indices.
            If True, will shuffle the input indices.
            If False, will create batches in the original order.

        edge_types: Sequence[Tuple[str]] (optional, default=None)
            The desired edge types for the subgraph.
            Defaults to all edges in the graph.

        directory: str (optional, default=new tempdir)
            The path of the directory to write samples to.
            Defaults to a new generated temporary directory.

        rank: int (optional, default=0)
            The rank of the current worker.  Should be provided
            when there are multiple workers.

        starting_batch_id: int (optional, default=0)
            The starting id for each batch.  Defaults to 0.
            Generally used when loading previously-sampled
            batches from disk.

        batches_per_partition: int (optional, default=100)
            The number of batches in each output partition.
            Defaults to 100.  Gets passed to the bulk
            sampler if there is one; otherwise, this argument
            is used to determine which files to read.
        """

        self.__feature_store = feature_store
        self.__graph_store = graph_store
        self.__rank = rank
        self.__next_batch = starting_batch_id
        self.__end_exclusive = starting_batch_id
        self.__batches_per_partition = batches_per_partition
        self.__starting_batch_id = starting_batch_id

        if isinstance(all_indices, int):
            # Will be loading from disk
            self.__num_batches = all_indices
            self.__directory = directory
            return

        if batch_size is None or batch_size < 1:
            raise ValueError("Batch size must be >= 1")

        self.__directory = tempfile.TemporaryDirectory(dir=directory)

        bulk_sampler = BulkSampler(
            batch_size,
            self.__directory.name,
            self.__graph_store._subgraph(edge_types),
            rank=rank,
            fanout_vals=num_neighbors,
            with_replacement=replace,
            batches_per_partition=self.__batches_per_partition**kwargs,
        )

        # Make sure indices are in cupy
        all_indices = cupy.asarray(all_indices)

        # Shuffle
        if shuffle:
            cupy.random.shuffle(all_indices)

        # Truncate if we can't evenly divide the input array
        stop = (len(all_indices) // batch_size) * batch_size
        all_indices = all_indices[:stop]

        # Split into batches
        all_indices = cupy.split(all_indices, len(all_indices) // batch_size)

        self.__num_batches = 0
        for batch_num, batch_i in enumerate(all_indices):
            self.__num_batches += 1
            bulk_sampler.add_batches(
                cudf.DataFrame(
                    {
                        "start": batch_i,
                        "batch": cupy.full(
                            batch_size, batch_num + starting_batch_id, dtype="int32"
                        ),
                    }
                ),
                start_col_name="start",
                batch_col_name="batch",
            )

        bulk_sampler.flush()

    def __next__(self):
        # Quit iterating if there are no batches left
        if self.__next_batch >= self.__num_batches + self.__starting_batch_id:
            raise StopIteration

        # Load the next set of sampling results if necessary
        if self.__next_batch >= self.__end_exclusive:
            # Read the next parquet file into memory
            dir_path = (
                self.__directory
                if isinstance(self.__directory, str)
                else self.__directory.name
            )
            rank_path = os.path.join(dir_path, f"rank={self.__rank}")

            parquet_path = os.path.join(
                rank_path,
                f"batch={self.__end_exclusive}"
                f"-{self.__end_exclusive + self.__batches_per_partition - 1}.parquet",
            )

            self.__end_exclusive += self.__batches_per_partition

            columns = {
                "sources": "int64",
                "destinations": "int64",
                # 'edge_id':'int64',
                "edge_type": "int32",
                "batch_id": "int32",
                # 'hop_id':'int32'
            }
            self.__data = cudf.read_parquet(parquet_path)
            self.__data = self.__data[list(columns.keys())].astype(columns)

        # Pull the next set of sampling results out of the dataframe in memory
        f = self.__data["batch_id"] == self.__next_batch
        sampler_output = _sampler_output_from_sampling_results(
            self.__data[f], self.__graph_store
        )

        # Get ready for next iteration
        # If there is no next iteration, make sure results are deleted
        self.__next_batch += 1
        if self.__next_batch >= self.__num_batches + self.__starting_batch_id:
            # Won't delete a non-temp dir (since it would just be deleting a string)
            del self.__directory

        # Get and return the sampled subgraph
        if isinstance(torch_geometric, MissingModule):
            noi_index, row_dict, col_dict, edge_dict = sampler_output["out"]
            return _filter_cugraph_store(
                self.__feature_store,
                self.__graph_store,
                noi_index,
                row_dict,
                col_dict,
                edge_dict,
            )
        else:
            return torch_geometric.loader.utils.filter_custom_store(
                self.__feature_store,
                self.__graph_store,
                sampler_output.node,
                sampler_output.row,
                sampler_output.col,
                sampler_output.edge,
            )

    def __iter__(self):
        return self


class EXPERIMENTAL__CuGraphNeighborLoader:
    def __init__(
        self,
        data: Union[CuGraphStore, Tuple[CuGraphStore, CuGraphStore]],
        input_nodes: Sequence,
        batch_size: int,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data: CuGraphStore or (CuGraphStore, CuGraphStore)
            The CuGraphStore or stores where the graph/feature data is held.

        batch_size: int
            The number of input nodes in each batch.

        input_nodes: Tensor
            The input nodes for *this* loader.  If there are multiple loaders,
            the appropriate split should be given for this loader.

        **kwargs: kwargs
            Keyword arguments to pass through for sampling.
            i.e. "shuffle", "fanout"
            See BulkSampleLoader.
        """

        # Allow passing in a feature store and graph store as a tuple, as
        # in the standard PyG API.  If only one is passed, it is assumed
        # it is behaving as both a graph store and a feature store.
        if isinstance(data, (list, tuple)):
            self.__feature_store, self.__graph_store = data
        else:
            self.__feature_store = data
            self.__graph_store = data

        self.__batch_size = batch_size
        self.__input_nodes = input_nodes
        self.inner_loader_args = kwargs

    def __iter__(self):
        return EXPERIMENTAL__BulkSampleLoader(
            self.__feature_store,
            self.__graph_store,
            self.__input_nodes,
            self.__batch_size,
            **self.inner_loader_args,
        )
