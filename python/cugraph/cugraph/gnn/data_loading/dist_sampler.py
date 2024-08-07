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

import os
import re
import warnings
from math import ceil
from functools import reduce

import pylibcugraph
import numpy as np
import cupy
import cudf

from typing import Union, List, Dict, Tuple, Iterator, Optional

from cugraph.utilities.utils import import_optional, MissingModule
from cugraph.gnn.comms import cugraph_comms_get_raft_handle

from cugraph.gnn.data_loading.bulk_sampler_io import create_df_from_disjoint_arrays

torch = MissingModule("torch")
TensorType = Union["torch.Tensor", cupy.ndarray, cudf.Series]


class DistSampleReader:
    def __init__(
        self,
        directory: str,
        *,
        format: str = "parquet",
        rank: Optional[int] = None,
        filelist=None,
    ):
        torch = import_optional("torch")

        self.__format = format
        self.__directory = directory

        if format != "parquet":
            raise ValueError("Invalid format (currently supported: 'parquet')")

        if filelist is None:
            files = os.listdir(directory)
            ex = re.compile(r"batch\=([0-9]+)\.([0-9]+)\-([0-9]+)\.([0-9]+)\.parquet")
            filematch = [ex.match(f) for f in files]
            filematch = [f for f in filematch if f]

            if rank is not None:
                filematch = [f for f in filematch if int(f[1]) == rank]

            batch_count = sum([int(f[4]) - int(f[2]) + 1 for f in filematch])
            filematch = sorted(filematch, key=lambda f: int(f[2]), reverse=True)

            self.__files = filematch
        else:
            self.__files = list(filelist)

        if rank is None:
            self.__batch_count = batch_count
        else:
            batch_count = torch.tensor([batch_count], device="cuda")
            torch.distributed.all_reduce(batch_count, torch.distributed.ReduceOp.MIN)
            self.__batch_count = int(batch_count)

    def __iter__(self):
        return self

    def __next__(self):
        torch = import_optional("torch")

        if len(self.__files) > 0:
            f = self.__files.pop()
            fname = f[0]
            start_inclusive = int(f[2])
            end_inclusive = int(f[4])

            if (end_inclusive - start_inclusive + 1) > self.__batch_count:
                end_inclusive = start_inclusive + self.__batch_count - 1
                self.__batch_count = 0
            else:
                self.__batch_count -= end_inclusive - start_inclusive + 1

            df = cudf.read_parquet(os.path.join(self.__directory, fname))
            tensors = {}
            for col in list(df.columns):
                s = df[col].dropna()
                if len(s) > 0:
                    tensors[col] = torch.as_tensor(s, device="cuda")
                df.drop(col, axis=1, inplace=True)

            return tensors, start_inclusive, end_inclusive

        raise StopIteration


class DistSampleWriter:
    def __init__(
        self,
        directory: str,
        *,
        batches_per_partition: int = 256,
        format: str = "parquet",
    ):
        """
        Parameters
        ----------
        directory: str (required)
            The directory where samples will be written.  This
            writer can only write to disk.
        batches_per_partition: int (optional, default=256)
            The number of batches to write in a single file.
        format: str (optional, default='parquet')
            The file format of the output files containing the
            sampled minibatches.  Currently, only parquet format
            is supported.
        """
        if format != "parquet":
            raise ValueError("Invalid format (currently supported: 'parquet')")

        self.__format = format
        self.__directory = directory
        self.__batches_per_partition = batches_per_partition

    @property
    def _format(self):
        return self.__format

    @property
    def _directory(self):
        return self.__directory

    @property
    def _batches_per_partition(self):
        return self.__batches_per_partition

    def get_reader(
        self, rank: int
    ) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        """
        Returns an iterator over sampled data.
        """

        # currently only disk reading is supported
        return DistSampleReader(self._directory, format=self._format, rank=rank)

    def __write_minibatches_coo(self, minibatch_dict):
        has_edge_ids = minibatch_dict["edge_id"] is not None
        has_edge_types = minibatch_dict["edge_type"] is not None
        has_weights = minibatch_dict["weight"] is not None

        if minibatch_dict["renumber_map"] is None:
            raise ValueError(
                "Distributed sampling without renumbering is not supported"
            )

        # Quit if there are no batches to write.
        if len(minibatch_dict["batch_id"]) == 0:
            return

        fanout_length = (len(minibatch_dict["label_hop_offsets"]) - 1) // len(
            minibatch_dict["batch_id"]
        )
        rank_batch_offset = minibatch_dict["batch_id"][0]

        for p in range(
            0, int(ceil(len(minibatch_dict["batch_id"]) / self.__batches_per_partition))
        ):
            partition_start = p * (self.__batches_per_partition)
            partition_end = (p + 1) * (self.__batches_per_partition)

            label_hop_offsets_array_p = minibatch_dict["label_hop_offsets"][
                partition_start * fanout_length : partition_end * fanout_length + 1
            ]

            batch_id_array_p = minibatch_dict["batch_id"][partition_start:partition_end]
            start_batch_id = batch_id_array_p[0] - rank_batch_offset

            start_ix, end_ix = label_hop_offsets_array_p[[0, -1]]
            majors_array_p = minibatch_dict["majors"][start_ix:end_ix]
            minors_array_p = minibatch_dict["minors"][start_ix:end_ix]
            edge_id_array_p = (
                minibatch_dict["edge_id"][start_ix:end_ix]
                if has_edge_ids
                else cupy.array([], dtype="int64")
            )
            edge_type_array_p = (
                minibatch_dict["edge_type"][start_ix:end_ix]
                if has_edge_types
                else cupy.array([], dtype="int32")
            )
            weight_array_p = (
                minibatch_dict["weight"][start_ix:end_ix]
                if has_weights
                else cupy.array([], dtype="float32")
            )

            # create the renumber map offsets
            renumber_map_offsets_array_p = minibatch_dict["renumber_map_offsets"][
                partition_start : partition_end + 1
            ]

            renumber_map_start_ix, renumber_map_end_ix = renumber_map_offsets_array_p[
                [0, -1]
            ]

            renumber_map_array_p = minibatch_dict["renumber_map"][
                renumber_map_start_ix:renumber_map_end_ix
            ]

            results_dataframe_p = create_df_from_disjoint_arrays(
                {
                    "majors": majors_array_p,
                    "minors": minors_array_p,
                    "map": renumber_map_array_p,
                    "label_hop_offsets": label_hop_offsets_array_p,
                    "weight": weight_array_p,
                    "edge_id": edge_id_array_p,
                    "edge_type": edge_type_array_p,
                    "renumber_map_offsets": renumber_map_offsets_array_p,
                }
            )

            end_batch_id = start_batch_id + len(batch_id_array_p) - 1
            rank = minibatch_dict["rank"] if "rank" in minibatch_dict else 0

            full_output_path = os.path.join(
                self.__directory,
                f"batch={rank:05d}.{start_batch_id:08d}-"
                f"{rank:05d}.{end_batch_id:08d}.parquet",
            )

            results_dataframe_p.to_parquet(
                full_output_path,
                compression=None,
                index=False,
                force_nullable_schema=True,
            )

    def __write_minibatches_csr(self, minibatch_dict):
        has_edge_ids = minibatch_dict["edge_id"] is not None
        has_edge_types = minibatch_dict["edge_type"] is not None
        has_weights = minibatch_dict["weight"] is not None

        if minibatch_dict["renumber_map"] is None:
            raise ValueError(
                "Distributed sampling without renumbering is not supported"
            )

        # Quit if there are no batches to write.
        if len(minibatch_dict["batch_id"]) == 0:
            return

        fanout_length = (len(minibatch_dict["label_hop_offsets"]) - 1) // len(
            minibatch_dict["batch_id"]
        )

        for p in range(
            0, int(ceil(len(minibatch_dict["batch_id"]) / self.__batches_per_partition))
        ):
            partition_start = p * (self.__batches_per_partition)
            partition_end = (p + 1) * (self.__batches_per_partition)

            label_hop_offsets_array_p = minibatch_dict["label_hop_offsets"][
                partition_start * fanout_length : partition_end * fanout_length + 1
            ]

            batch_id_array_p = minibatch_dict["batch_id"][partition_start:partition_end]
            start_batch_id = batch_id_array_p[0]

            # major offsets and minors
            (
                major_offsets_start_incl,
                major_offsets_end_incl,
            ) = label_hop_offsets_array_p[[0, -1]]

            start_ix, end_ix = minibatch_dict["major_offsets"][
                [major_offsets_start_incl, major_offsets_end_incl]
            ]

            major_offsets_array_p = minibatch_dict["major_offsets"][
                major_offsets_start_incl : major_offsets_end_incl + 1
            ]

            minors_array_p = minibatch_dict["minors"][start_ix:end_ix]
            edge_id_array_p = (
                minibatch_dict["edge_id"][start_ix:end_ix]
                if has_edge_ids
                else cupy.array([], dtype="int64")
            )
            edge_type_array_p = (
                minibatch_dict["edge_type"][start_ix:end_ix]
                if has_edge_types
                else cupy.array([], dtype="int32")
            )
            weight_array_p = (
                minibatch_dict["weight"][start_ix:end_ix]
                if has_weights
                else cupy.array([], dtype="float32")
            )

            # create the renumber map offsets
            renumber_map_offsets_array_p = minibatch_dict["renumber_map_offsets"][
                partition_start : partition_end + 1
            ]

            renumber_map_start_ix, renumber_map_end_ix = renumber_map_offsets_array_p[
                [0, -1]
            ]

            renumber_map_array_p = minibatch_dict["renumber_map"][
                renumber_map_start_ix:renumber_map_end_ix
            ]

            results_dataframe_p = create_df_from_disjoint_arrays(
                {
                    "major_offsets": major_offsets_array_p,
                    "minors": minors_array_p,
                    "map": renumber_map_array_p,
                    "label_hop_offsets": label_hop_offsets_array_p,
                    "weight": weight_array_p,
                    "edge_id": edge_id_array_p,
                    "edge_type": edge_type_array_p,
                    "renumber_map_offsets": renumber_map_offsets_array_p,
                }
            )

            end_batch_id = start_batch_id + len(batch_id_array_p) - 1
            rank = minibatch_dict["rank"] if "rank" in minibatch_dict else 0

            full_output_path = os.path.join(
                self.__directory,
                f"batch={rank:05d}.{start_batch_id:08d}-"
                f"{rank:05d}.{end_batch_id:08d}.parquet",
            )

            results_dataframe_p.to_parquet(
                full_output_path,
                compression=None,
                index=False,
                force_nullable_schema=True,
            )

    def write_minibatches(self, minibatch_dict):
        if (minibatch_dict["majors"] is not None) and (
            minibatch_dict["minors"] is not None
        ):
            self.__write_minibatches_coo(minibatch_dict)
        elif (minibatch_dict["major_offsets"] is not None) and (
            minibatch_dict["minors"] is not None
        ):
            self.__write_minibatches_csr(minibatch_dict)
        else:
            raise ValueError("invalid columns")


class DistSampler:
    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        writer: DistSampleWriter,
        local_seeds_per_call: int,
        retain_original_seeds: bool = False,
    ):
        """
        Parameters
        ----------
        graph: SGGraph or MGGraph (required)
            The pylibcugraph graph object that will be sampled.
        writer: DistSampleWriter (required)
            The writer responsible for writing samples to disk
            or, in the future, device or host memory.
        local_seeds_per_call: int
            The number of seeds on this rank this sampler will
            process in a single sampling call.  Batches will
            get split into multiple sampling calls based on
            this parameter.  This parameter must
            be the same across all ranks.  The total number
            of seeds processed per sampling call is this
            parameter times the world size. Subclasses should
            generally calculate the appropriate number of
            seeds.
        retain_original_seeds: bool (optional, default=False)
            Whether to retain the original seeds even if they
            do not appear in the output minibatch.  This will
            affect the output renumber map and CSR/CSC graph
            if applicable.
        """
        self.__graph = graph
        self.__writer = writer
        self.__local_seeds_per_call = local_seeds_per_call
        self.__handle = None
        self.__retain_original_seeds = retain_original_seeds

    def get_reader(self) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        """
        Returns an iterator over sampled data.
        """
        torch = import_optional("torch")
        rank = torch.distributed.get_rank() if self.is_multi_gpu else None
        return self.__writer.get_reader(rank)

    def sample_batches(
        self,
        seeds: TensorType,
        batch_ids: TensorType,
        random_state: int = 0,
        assume_equal_input_size: bool = False,
    ) -> Dict[str, TensorType]:
        """
        For a single call group of seeds and associated batch ids, performs
        sampling.

        Parameters
        ----------
        seeds: TensorType
            Input seeds for a single call group (node ids).
        batch_ids: TensorType
            The batch id for each seed.
        random_state: int
            The random seed to use for sampling.
        assume_equal_input_size: bool
            If True, will assume all ranks have the same number of inputs,
            and will skip the synchronization/gather steps to check for
            and handle uneven inputs.

        Returns
        -------
        A dictionary containing the sampling outputs (majors, minors, map, etc.)
        """
        raise NotImplementedError("Must be implemented by subclass")

    def get_label_list_and_output_rank(
        self, local_label_list: TensorType, assume_equal_input_size: bool = False
    ):
        """
        Computes the label list and output rank mapping for
        the list of labels (batch ids).
        Subclasses may override this as needed depending on their
        memory and compute constraints.

        Parameters
        ----------
        local_label_list: TensorType
            The list of unique labels on this rank.
        assume_equal_input_size: bool
            If True, assumes that all ranks have the same number of inputs (batches)
            and skips some synchronization/gathering accordingly.

        Returns
        -------
        label_list: TensorType
            The global label list containing all labels used across ranks.
        label_to_output_comm_rank: TensorType
            The global mapping of labels to ranks.
        """
        torch = import_optional("torch")

        world_size = torch.distributed.get_world_size()

        if assume_equal_input_size:
            num_batches = len(local_label_list) * world_size
            label_list = torch.empty((num_batches,), dtype=torch.int32, device="cuda")
            w = torch.distributed.all_gather_into_tensor(
                label_list, local_label_list, async_op=True
            )

            label_to_output_comm_rank = torch.concat(
                [
                    torch.full(
                        (len(local_label_list),), r, dtype=torch.int32, device="cuda"
                    )
                    for r in range(world_size)
                ]
            )
        else:
            num_batches = torch.tensor(
                [len(local_label_list)], device="cuda", dtype=torch.int64
            )
            num_batches_all_ranks = torch.empty(
                (world_size,), device="cuda", dtype=torch.int64
            )
            torch.distributed.all_gather_into_tensor(num_batches_all_ranks, num_batches)

            label_list = [
                torch.empty((n,), dtype=torch.int32, device="cuda")
                for n in num_batches_all_ranks
            ]
            w = torch.distributed.all_gather(
                label_list, local_label_list, async_op=True
            )

            label_to_output_comm_rank = torch.concat(
                [
                    torch.full((num_batches_r,), r, device="cuda", dtype=torch.int32)
                    for r, num_batches_r in enumerate(num_batches_all_ranks)
                ]
            )

        w.wait()
        if isinstance(label_list, list):
            label_list = torch.concat(label_list)
        return label_list, label_to_output_comm_rank

    def get_start_batch_offset(
        self, local_num_batches: int, assume_equal_input_size: bool = False
    ) -> Tuple[int, bool]:
        """
        Gets the starting batch offset to ensure each rank's set of batch ids is
        disjoint.

        Parameters
        ----------
        local_num_batches: int
            The number of batches for this rank.
        assume_equal_input_size: bool
            Whether to assume all ranks have the same number of batches.

        Returns
        -------
        Tuple[int, bool]
            The starting batch offset (int)
            and whether the input sizes on each rank are equal (bool).

        """
        torch = import_optional("torch")

        input_size_is_equal = True
        if self.is_multi_gpu:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            if assume_equal_input_size:
                t = torch.full(
                    (world_size,), local_num_batches, dtype=torch.int64, device="cuda"
                )
            else:
                t = torch.empty((world_size,), dtype=torch.int64, device="cuda")
                local_size = torch.tensor(
                    [local_num_batches], dtype=torch.int64, device="cuda"
                )

                torch.distributed.all_gather_into_tensor(t, local_size)
                if (t != local_size).any():
                    input_size_is_equal = False
                    if rank == 0:
                        warnings.warn(
                            "Not all ranks received the same number of batches. "
                            "This might cause your training loop to hang "
                            "due to uneven inputs."
                        )

            return (0 if rank == 0 else t.cumsum(dim=0)[rank - 1], input_size_is_equal)
        else:
            return 0, input_size_is_equal

    def sample_from_nodes(
        self,
        nodes: TensorType,
        *,
        batch_size: int = 16,
        random_state: int = 62,
        assume_equal_input_size: bool = False,
    ):
        """
        Performs node-based sampling.  Accepts a list of seed nodes, and batch size.
        Splits the seed list into batches, then divides the batches into call groups
        based on the number of seeds per call this sampler was set to use.
        Then calls sample_batches for each call group and writes the result using
        the writer associated with this sampler.

        Parameters
        ----------
        nodes: TensorType
            Input seeds (node ids).
        batch_size: int
            The size of each batch.
        random_state: int
            The random seed to use for sampling.
        """
        torch = import_optional("torch")

        nodes = torch.as_tensor(nodes, device="cuda")

        batches_per_call = self._local_seeds_per_call // batch_size
        actual_seeds_per_call = batches_per_call * batch_size

        # Split the input seeds into call groups.  Each call group
        # corresponds to one sampling call.  A call group contains
        # many batches.
        num_seeds = len(nodes)
        nodes_call_groups = torch.split(nodes, actual_seeds_per_call)

        local_num_batches = int(ceil(num_seeds / batch_size))
        batch_id_start, input_size_is_equal = self.get_start_batch_offset(
            local_num_batches, assume_equal_input_size=assume_equal_input_size
        )

        # Need to add empties to the list of call groups to handle the case
        # where not all nodes have the same number of call groups.  This
        # prevents a hang since we need all ranks to make the same number
        # of calls.
        if not input_size_is_equal:
            num_call_groups = torch.tensor(
                [len(nodes_call_groups)], device="cuda", dtype=torch.int32
            )
            torch.distributed.all_reduce(
                num_call_groups, op=torch.distributed.ReduceOp.MAX
            )
            nodes_call_groups = list(nodes_call_groups) + (
                [torch.tensor([], dtype=nodes.dtype, device="cuda")]
                * (int(num_call_groups) - len(nodes_call_groups))
            )

        # Make a call to sample_batches for each call group
        for i, current_seeds in enumerate(nodes_call_groups):
            current_batches = torch.arange(
                batch_id_start + i * batches_per_call,
                batch_id_start
                + i * batches_per_call
                + int(ceil(len(current_seeds)))
                + 1,
                device="cuda",
                dtype=torch.int32,
            )

            current_batches = current_batches.repeat_interleave(batch_size)[
                : len(current_seeds)
            ]

            print(
                current_seeds,
                current_batches,
                flush=True,
            )
            minibatch_dict = self.sample_batches(
                seeds=current_seeds,
                batch_ids=current_batches,
                random_state=random_state,
                assume_equal_input_size=input_size_is_equal,
            )
            self.__writer.write_minibatches(minibatch_dict)

    @property
    def is_multi_gpu(self):
        return isinstance(self.__graph, pylibcugraph.MGGraph)

    @property
    def _local_seeds_per_call(self):
        return self.__local_seeds_per_call

    @property
    def _graph(self):
        return self.__graph

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
    def _retain_original_seeds(self):
        return self.__retain_original_seeds


class NeighborSampler(DistSampler):
    # Number of vertices in the output minibatch, based
    # on benchmarking.
    BASE_VERTICES_PER_BYTE = 0.1107662486009992

    # Default number of seeds if the output minibatch
    # size can't be estimated.
    UNKNOWN_VERTICES_DEFAULT = 32768

    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        writer: DistSampleWriter,
        *,
        local_seeds_per_call: Optional[int] = None,
        retain_original_seeds: bool = False,
        fanout: List[int] = [-1],
        prior_sources_behavior: str = "exclude",
        deduplicate_sources: bool = True,
        compression: str = "COO",
        compress_per_hop: bool = False,
        with_replacement: bool = False,
        biased: bool = False,
    ):
        self.__fanout = fanout
        self.__prior_sources_behavior = prior_sources_behavior
        self.__deduplicate_sources = deduplicate_sources
        self.__compress_per_hop = compress_per_hop
        self.__compression = compression
        self.__with_replacement = with_replacement

        # It is currently required that graphs are weighted for biased
        # sampling.  So setting the function here is safe.  In the future,
        # if libcugraph allows setting a new attribute, this API might
        # change.
        self.__func = (
            pylibcugraph.biased_neighbor_sample
            if biased
            else pylibcugraph.uniform_neighbor_sample
        )

        super().__init__(
            graph,
            writer,
            local_seeds_per_call=self.__calc_local_seeds_per_call(local_seeds_per_call),
            retain_original_seeds=retain_original_seeds,
        )

    def __calc_local_seeds_per_call(self, local_seeds_per_call: Optional[int] = None):
        torch = import_optional("torch")

        if local_seeds_per_call is None:
            if len([x for x in self.__fanout if x <= 0]) > 0:
                return NeighborSampler.UNKNOWN_VERTICES_DEFAULT

            total_memory = torch.cuda.get_device_properties(0).total_memory
            fanout_prod = reduce(lambda x, y: x * y, self.__fanout)
            return int(
                NeighborSampler.BASE_VERTICES_PER_BYTE * total_memory / fanout_prod
            )

        return local_seeds_per_call

    def sample_batches(
        self,
        seeds: TensorType,
        batch_ids: TensorType,
        random_state: int = 0,
        assume_equal_input_size: bool = False,
    ) -> Dict[str, TensorType]:
        torch = import_optional("torch")
        if self.is_multi_gpu:
            rank = torch.distributed.get_rank()

            batch_ids = batch_ids.to(device="cuda", dtype=torch.int32)
            local_label_list = torch.unique(batch_ids)

            label_list, label_to_output_comm_rank = self.get_label_list_and_output_rank(
                local_label_list, assume_equal_input_size=assume_equal_input_size
            )

            if self._retain_original_seeds:
                label_offsets = torch.concat(
                    [
                        torch.searchsorted(batch_ids, local_label_list),
                        torch.tensor(
                            [batch_ids.shape[0]], device="cuda", dtype=torch.int64
                        ),
                    ]
                )
            else:
                label_offsets = None

            sampling_results_dict = self.__func(
                self._resource_handle,
                self._graph,
                start_list=cupy.asarray(seeds),
                batch_id_list=cupy.asarray(batch_ids),
                label_list=cupy.asarray(label_list),
                label_to_output_comm_rank=cupy.asarray(label_to_output_comm_rank),
                h_fan_out=np.array(self.__fanout, dtype="int32"),
                with_replacement=self.__with_replacement,
                do_expensive_check=True,
                with_edge_properties=False,
                random_state=random_state + rank,
                prior_sources_behavior=self.__prior_sources_behavior,
                deduplicate_sources=self.__deduplicate_sources,
                return_hops=True,
                renumber=True,
                compression=self.__compression,
                compress_per_hop=self.__compress_per_hop,
                retain_seeds=self._retain_original_seeds,
                label_offsets=None
                if label_offsets is None
                else cupy.asarray(label_offsets),
                return_dict=True,
            )
            sampling_results_dict["rank"] = rank
        else:
            if self._retain_original_seeds:
                batch_ids = batch_ids.to(device="cuda", dtype=torch.int32)
                local_label_list = torch.unique(batch_ids)
                label_offsets = torch.concat(
                    [
                        torch.searchsorted(batch_ids, local_label_list),
                        torch.tensor(
                            [batch_ids.shape[0]], device="cuda", dtype=torch.int64
                        ),
                    ]
                )
            else:
                label_offsets = None

            sampling_results_dict = self.__func(
                self._resource_handle,
                self._graph,
                start_list=cupy.asarray(seeds),
                batch_id_list=cupy.asarray(batch_ids),
                h_fan_out=np.array(self.__fanout, dtype="int32"),
                with_replacement=self.__with_replacement,
                do_expensive_check=False,
                with_edge_properties=True,
                random_state=random_state,
                prior_sources_behavior=self.__prior_sources_behavior,
                deduplicate_sources=self.__deduplicate_sources,
                return_hops=True,
                renumber=True,
                compression=self.__compression,
                compress_per_hop=self.__compress_per_hop,
                retain_seeds=self._retain_original_seeds,
                label_offsets=None
                if label_offsets is None
                else cupy.asarray(label_offsets),
                return_dict=True,
            )

        return sampling_results_dict
