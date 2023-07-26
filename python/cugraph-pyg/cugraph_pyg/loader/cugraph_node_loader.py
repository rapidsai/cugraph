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
import re

import cupy
import cudf

from cugraph.experimental.gnn import BulkSampler
from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.loader.filter import _filter_cugraph_store
from cugraph_pyg.sampler.cugraph_sampler import _sampler_output_from_sampling_results

from typing import Union, Tuple, Sequence, List, Dict

torch_geometric = import_optional("torch_geometric")
InputNodes = (
    Sequence
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.typing.InputNodes
)


class EXPERIMENTAL__BulkSampleLoader:

    __ex_parquet_file = re.compile(r"batch=([0-9]+)\-([0-9]+)\.parquet")

    def __init__(
        self,
        feature_store: CuGraphStore,
        graph_store: CuGraphStore,
        input_nodes: InputNodes = None,
        batch_size: int = 0,
        shuffle: bool = False,
        edge_types: Sequence[Tuple[str]] = None,
        directory: Union[str, tempfile.TemporaryDirectory] = None,
        input_files: List[str] = None,
        starting_batch_id: int = 0,
        batches_per_partition: int = 100,
        # Sampler args
        num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]] = None,
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

        input_nodes: InputNodes
            The input nodes associated with this sampler.
            If None, this loader will load batches
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

        input_files: List[str] (optional, default=None)
            The input files to read from the directory containing
            samples.  This argument is only used when loading
            alread-sampled batches from disk.

        starting_batch_id: int (optional, default=0)
            The starting id for each batch.  Defaults to 0.

        batches_per_partition: int (optional, default=100)
            The number of batches in each output partition.
            Defaults to 100.  Gets passed to the bulk
            sampler if there is one; otherwise, this argument
            is used to determine which files to read.

        num_neighbors: Union[List[int],
                 Dict[Tuple[str, str, str], List[int]]] (required)
            The number of neighbors to sample for each node in each iteration.
            If an entry is set to -1, all neighbors will be included.
            In heterogeneous graphs, may also take in a dictionary denoting
            the number of neighbors to sample for each individual edge type.

            Note: in cuGraph, only one value of num_neighbors is currently supported.
            Passing in a dictionary will result in an exception.
        """

        self.__feature_store = feature_store
        self.__graph_store = graph_store
        self.__next_batch = -1
        self.__end_exclusive = -1
        self.__batches_per_partition = batches_per_partition
        self.__starting_batch_id = starting_batch_id

        if input_nodes is None:
            # Will be loading from disk
            self.__num_batches = input_nodes
            self.__directory = directory
            if input_files is None:
                if isinstance(self.__directory, str):
                    self.__input_files = iter(os.listdir(self.__directory))
                else:
                    self.__input_files = iter(os.listdir(self.__directory.name))
            else:
                self.__input_files = iter(input_files)
            return

        input_type, input_nodes = torch_geometric.loader.utils.get_input_nodes(
            (feature_store, graph_store), input_nodes
        )
        if input_type is not None:
            input_nodes = graph_store._get_sample_from_vertex_groups(
                {input_type: input_nodes}
            )

        if batch_size is None or batch_size < 1:
            raise ValueError("Batch size must be >= 1")

        self.__directory = tempfile.TemporaryDirectory(dir=directory)

        if isinstance(num_neighbors, dict):
            raise ValueError("num_neighbors dict is currently unsupported!")

        renumber = (
            True
            if (
                (len(self.__graph_store.node_types) == 1)
                and (len(self.__graph_store.edge_types) == 1)
            )
            else False
        )

        bulk_sampler = BulkSampler(
            batch_size,
            self.__directory.name,
            self.__graph_store._subgraph(edge_types),
            fanout_vals=num_neighbors,
            with_replacement=replace,
            batches_per_partition=self.__batches_per_partition,
            renumber=renumber,
            **kwargs,
        )

        # Make sure indices are in cupy
        input_nodes = cupy.asarray(input_nodes)

        # Shuffle
        if shuffle:
            cupy.random.shuffle(input_nodes)

        # Truncate if we can't evenly divide the input array
        stop = (len(input_nodes) // batch_size) * batch_size
        input_nodes = input_nodes[:stop]

        # Split into batches
        input_nodes = cupy.split(input_nodes, len(input_nodes) // batch_size)

        self.__num_batches = 0
        for batch_num, batch_i in enumerate(input_nodes):
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
        self.__input_files = iter(os.listdir(self.__directory.name))

    def __next__(self):
        # Load the next set of sampling results if necessary
        if self.__next_batch >= self.__end_exclusive:
            if self.__directory is None:
                raise StopIteration

            # Read the next parquet file into memory
            dir_path = (
                self.__directory
                if isinstance(self.__directory, str)
                else self.__directory.name
            )

            # Will raise StopIteration if there are no files left
            try:
                fname = next(self.__input_files)
            except StopIteration as ex:
                # Won't delete a non-temp dir (since it would just be deleting a string)
                del self.__directory
                self.__directory = None
                raise StopIteration(ex)

            m = self.__ex_parquet_file.match(fname)
            if m is None:
                raise ValueError(f"Invalid parquet filename {fname}")

            self.__start_inclusive, end_inclusive = [int(g) for g in m.groups()]
            self.__next_batch = self.__start_inclusive
            self.__end_exclusive = end_inclusive + 1

            parquet_path = os.path.join(
                dir_path,
                fname,
            )

            columns = {
                "sources": "int64",
                "destinations": "int64",
                # 'edge_id':'int64',
                "edge_type": "int32",
                "batch_id": "int32",
                "hop_id": "int32",
            }

            raw_sample_data = cudf.read_parquet(parquet_path)
            if "map" in raw_sample_data.columns:
                self.__renumber_map = raw_sample_data["map"]
                raw_sample_data.drop("map", axis=1, inplace=True)
            else:
                self.__renumber_map = None

            self.__data = raw_sample_data[list(columns.keys())].astype(columns)
            self.__data.dropna(inplace=True)

        # Pull the next set of sampling results out of the dataframe in memory
        f = self.__data["batch_id"] == self.__next_batch
        if self.__renumber_map is not None:
            i = self.__next_batch - self.__start_inclusive
            ix = self.__renumber_map.iloc[[i, i + 1]]
            ix_start, ix_end = ix.iloc[0], ix.iloc[1]
            current_renumber_map = self.__renumber_map.iloc[ix_start:ix_end]
            if len(current_renumber_map) != ix_end - ix_start:
                raise ValueError("invalid renumber map")
        else:
            current_renumber_map = None

        sampler_output = _sampler_output_from_sampling_results(
            self.__data[f], current_renumber_map, self.__graph_store
        )

        # Get ready for next iteration
        self.__next_batch += 1

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
            out = torch_geometric.loader.utils.filter_custom_store(
                self.__feature_store,
                self.__graph_store,
                sampler_output.node,
                sampler_output.row,
                sampler_output.col,
                sampler_output.edge,
            )

            return out

    def __iter__(self):
        return self


class EXPERIMENTAL__CuGraphNeighborLoader:
    def __init__(
        self,
        data: Union[CuGraphStore, Tuple[CuGraphStore, CuGraphStore]],
        input_nodes: Union[InputNodes, int] = None,
        batch_size: int = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        data: CuGraphStore or (CuGraphStore, CuGraphStore)
            The CuGraphStore or stores where the graph/feature data is held.

        batch_size: int (required)
            The number of input nodes in each batch.

        input_nodes: Union[InputNodes, int] (required)
            The input nodes associated with this sampler.

        **kwargs: kwargs
            Keyword arguments to pass through for sampling.
            i.e. "shuffle", "fanout"
            See BulkSampleLoader.
        """

        if input_nodes is None:
            raise ValueError("input_nodes is required")
        if batch_size is None:
            raise ValueError("batch_size is required")

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
        self.current_loader = EXPERIMENTAL__BulkSampleLoader(
            self.__feature_store,
            self.__graph_store,
            self.__input_nodes,
            self.__batch_size,
            **self.inner_loader_args,
        )

        return self.current_loader
