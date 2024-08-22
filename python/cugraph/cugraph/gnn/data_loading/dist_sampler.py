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
from math import ceil
from functools import reduce

import pylibcugraph
import numpy as np
import cupy
import cudf

from typing import Union, List, Dict, Tuple, Iterator, Optional

from cugraph.utilities.utils import import_optional, MissingModule
from cugraph.gnn.comms import cugraph_comms_get_raft_handle


from cugraph.gnn.data_loading.dist_io import BufferedSampleReader
from cugraph.gnn.data_loading.dist_io import DistSampleWriter

torch = MissingModule("torch")
TensorType = Union["torch.Tensor", cupy.ndarray, cudf.Series]


class DistSampler:
    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        writer: Optional[DistSampleWriter],
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
            or; if None, then samples will be written to memory
            instead.
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

    def __sample_from_nodes_func(
        self,
        call_id: int,
        current_seeds: "torch.Tensor",
        batch_id_start: int,
        batch_size: int,
        batches_per_call: int,
        random_state: int,
        assume_equal_input_size: bool,
    ) -> Union[None, Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]]:
        torch = import_optional("torch")

        current_batches = torch.arange(
            batch_id_start + call_id * batches_per_call,
            batch_id_start
            + call_id * batches_per_call
            + int(ceil(len(current_seeds)))
            + 1,
            device="cuda",
            dtype=torch.int32,
        )

        current_batches = current_batches.repeat_interleave(batch_size)[
            : len(current_seeds)
        ]

        minibatch_dict = self.sample_batches(
            seeds=current_seeds,
            batch_ids=current_batches,
            random_state=random_state,
            assume_equal_input_size=assume_equal_input_size,
        )

        if self.__writer is None:
            return iter([(minibatch_dict, current_batches[0], current_batches[-1])])
        else:
            self.__writer.write_minibatches(minibatch_dict)
            return None

    def sample_from_nodes(
        self,
        nodes: TensorType,
        *,
        batch_size: int = 16,
        random_state: int = 62,
        assume_equal_input_size: bool = False,
    ) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
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

        sample_args = (
            batch_id_start,
            batch_size,
            batches_per_call,
            random_state,
            input_size_is_equal,
        )

        if self.__writer is None:
            # Buffered sampling
            return BufferedSampleReader(
                nodes_call_groups, self.__sample_from_nodes_func, *sample_args
            )
        else:
            # Unbuffered sampling
            for i, current_seeds in enumerate(nodes_call_groups):
                self.__sample_from_nodes_func(
                    i,
                    current_seeds,
                    *sample_args,
                )

            # Return a reader that points to the stored samples
            rank = torch.distributed.get_rank() if self.is_multi_gpu else None
            return self.__writer.get_reader(rank)

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
        # TODO allow func to be a call to a future remote sampling API
        # if the provided graph is in another process (rapidsai/cugraph#4623).
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
                with_edge_properties=True,
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
