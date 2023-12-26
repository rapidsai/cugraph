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
import warnings

import cupy
import cudf

from cugraph.experimental.gnn import BulkSampler
from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.sampler.cugraph_sampler import (
    _sampler_output_from_sampling_results_heterogeneous,
    _sampler_output_from_sampling_results_homogeneous_csr,
    _sampler_output_from_sampling_results_homogeneous_coo,
    filter_cugraph_store_csc,
)

from typing import Union, Tuple, Sequence, List, Dict

torch_geometric = import_optional("torch_geometric")
torch = import_optional("torch")
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
        *,
        shuffle: bool = False,
        drop_last: bool = True,
        edge_types: Sequence[Tuple[str]] = None,
        directory: Union[str, tempfile.TemporaryDirectory] = None,
        input_files: List[str] = None,
        starting_batch_id: int = 0,
        batches_per_partition: int = 100,
        # Sampler args
        num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]] = None,
        replace: bool = True,
        compression: str = "COO",
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

        self._total_read_time = 0.0
        self._total_convert_time = 0.0
        self._total_feature_time = 0.0

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

        input_node_info = torch_geometric.loader.utils.get_input_nodes(
            (feature_store, graph_store),
            input_nodes,
            input_id=None,
        )

        # PyG 2.4
        if len(input_node_info) == 2:
            input_type, input_nodes = input_node_info
        # PyG 2.5
        elif len(input_node_info) == 3:
            input_type, input_nodes, input_id = input_node_info
        # Invalid
        else:
            raise ValueError("Invalid output from get_input_nodes")

        if input_type is not None:
            input_nodes = graph_store._get_sample_from_vertex_groups(
                {input_type: input_nodes}
            )

        if batch_size is None or batch_size < 1:
            raise ValueError("Batch size must be >= 1")

        self.__directory = (
            tempfile.TemporaryDirectory() if directory is None else directory
        )

        if isinstance(num_neighbors, dict):
            raise ValueError("num_neighbors dict is currently unsupported!")

        if "renumber" in kwargs:
            warnings.warn(
                "Setting renumbering manually could result in invalid output,"
                " please ensure you intended to do this."
            )
            renumber = kwargs.pop("renumber")
        else:
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
            self.__directory
            if isinstance(self.__directory, str)
            else self.__directory.name,
            self.__graph_store._subgraph(edge_types),
            fanout_vals=num_neighbors,
            with_replacement=replace,
            batches_per_partition=self.__batches_per_partition,
            renumber=renumber,
            use_legacy_names=False,
            deduplicate_sources=True,
            prior_sources_behavior="exclude",
            include_hop_column=(compression == "COO"),
            **kwargs,
        )

        # Make sure indices are in cupy
        input_nodes = cupy.asarray(input_nodes)

        # Shuffle
        if shuffle:
            cupy.random.shuffle(input_nodes)

        # Truncate if we can't evenly divide the input array
        stop = (len(input_nodes) // batch_size) * batch_size
        input_nodes, remainder = cupy.array_split(input_nodes, [stop])

        # Split into batches
        input_nodes = cupy.split(input_nodes, max(len(input_nodes) // batch_size, 1))

        if not drop_last:
            input_nodes.append(remainder)

        self.__num_batches = 0
        for batch_num, batch_i in enumerate(input_nodes):
            batch_len = len(batch_i)
            if batch_len > 0:
                self.__num_batches += 1
                bulk_sampler.add_batches(
                    cudf.DataFrame(
                        {
                            "start": batch_i,
                            "batch": cupy.full(
                                batch_len, batch_num + starting_batch_id, dtype="int32"
                            ),
                        }
                    ),
                    start_col_name="start",
                    batch_col_name="batch",
                )

        bulk_sampler.flush()
        self.__input_files = iter(
            os.listdir(
                self.__directory
                if isinstance(self.__directory, str)
                else self.__directory.name
            )
        )

    def __next__(self):
        from time import perf_counter

        start_time_read_data = perf_counter()

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

            raw_sample_data = cudf.read_parquet(parquet_path)

            if "map" in raw_sample_data.columns:
                if "renumber_map_offsets" not in raw_sample_data.columns:
                    num_batches = end_inclusive - self.__start_inclusive + 1

                    map_end = raw_sample_data["map"].iloc[num_batches]

                    map = torch.as_tensor(
                        raw_sample_data["map"].iloc[0:map_end], device="cuda"
                    )
                    raw_sample_data.drop("map", axis=1, inplace=True)

                    self.__renumber_map_offsets = map[0 : num_batches + 1] - map[0]
                    self.__renumber_map = map[num_batches + 1 :]
                else:
                    self.__renumber_map = raw_sample_data["map"]
                    self.__renumber_map_offsets = raw_sample_data[
                        "renumber_map_offsets"
                    ]
                    raw_sample_data.drop(
                        columns=["map", "renumber_map_offsets"], inplace=True
                    )

                    self.__renumber_map.dropna(inplace=True)
                    self.__renumber_map = torch.as_tensor(
                        self.__renumber_map, device="cuda"
                    )

                    self.__renumber_map_offsets.dropna(inplace=True)
                    self.__renumber_map_offsets = torch.as_tensor(
                        self.__renumber_map_offsets, device="cuda"
                    )

            else:
                self.__renumber_map = None

            self.__data = raw_sample_data
            self.__coo = "majors" in self.__data.columns
            if self.__coo:
                self.__data.dropna(inplace=True)

            if (
                len(self.__graph_store.edge_types) == 1
                and len(self.__graph_store.node_types) == 1
            ):
                if self.__coo:
                    group_cols = ["batch_id", "hop_id"]
                    self.__data_index = self.__data.groupby(
                        group_cols, as_index=True
                    ).agg({"majors": "max", "minors": "max"})
                    self.__data_index.rename(
                        columns={"majors": "src_max", "minors": "dst_max"},
                        inplace=True,
                    )
                    self.__data_index = self.__data_index.to_dict(orient="index")
                else:
                    self.__data_index = None

                    self.__label_hop_offsets = self.__data["label_hop_offsets"]
                    self.__data.drop(columns=["label_hop_offsets"], inplace=True)
                    self.__label_hop_offsets.dropna(inplace=True)
                    self.__label_hop_offsets = torch.as_tensor(
                        self.__label_hop_offsets, device="cuda"
                    )
                    self.__label_hop_offsets -= self.__label_hop_offsets[0].clone()

                    self.__major_offsets = self.__data["major_offsets"]
                    self.__data.drop(columns="major_offsets", inplace=True)
                    self.__major_offsets.dropna(inplace=True)
                    self.__major_offsets = torch.as_tensor(
                        self.__major_offsets, device="cuda"
                    )
                    self.__major_offsets -= self.__major_offsets[0].clone()

                    self.__minors = self.__data["minors"]
                    self.__data.drop(columns="minors", inplace=True)
                    self.__minors.dropna(inplace=True)
                    self.__minors = torch.as_tensor(self.__minors, device="cuda")

                    num_batches = self.__end_exclusive - self.__start_inclusive
                    offsets_len = len(self.__label_hop_offsets) - 1
                    if offsets_len % num_batches != 0:
                        raise ValueError("invalid label-hop offsets")
                    self.__fanout_length = int(offsets_len / num_batches)

        end_time_read_data = perf_counter()
        self._total_read_time += end_time_read_data - start_time_read_data

        # Pull the next set of sampling results out of the dataframe in memory
        if self.__coo:
            f = self.__data["batch_id"] == self.__next_batch
        if self.__renumber_map is not None:
            i = self.__next_batch - self.__start_inclusive

            # this should avoid d2h copy
            current_renumber_map = self.__renumber_map[
                self.__renumber_map_offsets[i] : self.__renumber_map_offsets[i + 1]
            ]

        else:
            current_renumber_map = None

        start_time_convert = perf_counter()
        # Get and return the sampled subgraph
        if (
            len(self.__graph_store.edge_types) == 1
            and len(self.__graph_store.node_types) == 1
        ):
            if self.__coo:
                sampler_output = _sampler_output_from_sampling_results_homogeneous_coo(
                    self.__data[f],
                    current_renumber_map,
                    self.__graph_store,
                    self.__data_index,
                    self.__next_batch,
                )
            else:
                i = (self.__next_batch - self.__start_inclusive) * self.__fanout_length
                current_label_hop_offsets = self.__label_hop_offsets[
                    i : i + self.__fanout_length + 1
                ]

                current_major_offsets = self.__major_offsets[
                    current_label_hop_offsets[0] : (current_label_hop_offsets[-1] + 1)
                ]

                current_minors = self.__minors[
                    current_major_offsets[0] : current_major_offsets[-1]
                ]

                sampler_output = _sampler_output_from_sampling_results_homogeneous_csr(
                    current_major_offsets,
                    current_minors,
                    current_renumber_map,
                    self.__graph_store,
                    current_label_hop_offsets,
                    self.__data_index,
                    self.__next_batch,
                )
        else:
            sampler_output = _sampler_output_from_sampling_results_heterogeneous(
                self.__data[f], current_renumber_map, self.__graph_store
            )

        # Get ready for next iteration
        self.__next_batch += 1

        end_time_convert = perf_counter()
        self._total_convert_time += end_time_convert - start_time_convert

        start_time_feature = perf_counter()
        # Create a PyG HeteroData object, loading the required features
        if self.__coo:
            out = torch_geometric.loader.utils.filter_custom_hetero_store(
                self.__feature_store,
                self.__graph_store,
                sampler_output.node,
                sampler_output.row,
                sampler_output.col,
                sampler_output.edge,
            )
        else:
            out = filter_cugraph_store_csc(
                self.__feature_store,
                self.__graph_store,
                sampler_output.node,
                sampler_output.row,
                sampler_output.col,
                sampler_output.edge,
            )

        # Account for CSR format in cuGraph vs. CSC format in PyG
        if self.__coo and self.__graph_store.order == "CSC":
            for edge_type in out.edge_index_dict:
                out[edge_type].edge_index = out[edge_type].edge_index.flip(dims=[0])

        out.set_value_dict("num_sampled_nodes", sampler_output.num_sampled_nodes)
        out.set_value_dict("num_sampled_edges", sampler_output.num_sampled_edges)

        end_time_feature = perf_counter()
        self._total_feature_time = end_time_feature - start_time_feature

        return out

    @property
    def _starting_batch_id(self):
        return self.__starting_batch_id

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

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    def __iter__(self):
        self.current_loader = EXPERIMENTAL__BulkSampleLoader(
            self.__feature_store,
            self.__graph_store,
            self.__input_nodes,
            self.__batch_size,
            **self.inner_loader_args,
        )

        return self.current_loader
