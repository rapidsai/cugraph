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
        current_seeds_and_ix: Tuple["torch.Tensor", "torch.Tensor"],
        batch_id_start: int,
        batch_size: int,
        batches_per_call: int,
        random_state: int,
        assume_equal_input_size: bool,
    ) -> Union[None, Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]]:
        torch = import_optional("torch")

        current_seeds, current_ix = current_seeds_and_ix

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

        # do qr division to get the number of batch_size batches and the
        # size of the last batch
        num_full, last_count = divmod(len(current_seeds), batch_size)
        input_offsets = torch.concatenate(
            [
                torch.tensor([0], device="cuda", dtype=torch.int64),
                torch.full((num_full,), batch_size, device="cuda", dtype=torch.int64),
                torch.tensor([last_count], device="cuda", dtype=torch.int64)
                if last_count > 0
                else torch.tensor([], device="cuda", dtype=torch.int64),
            ]
        ).cumsum(-1)

        minibatch_dict = self.sample_batches(
            seeds=current_seeds,
            batch_ids=current_batches,
            random_state=random_state,
            assume_equal_input_size=assume_equal_input_size,
        )
        minibatch_dict["input_index"] = current_ix.cuda()
        minibatch_dict["input_offsets"] = input_offsets

        if self.__writer is None:
            # rename renumber_map -> map to match unbuffered format
            minibatch_dict["map"] = minibatch_dict["renumber_map"]
            del minibatch_dict["renumber_map"]
            minibatch_dict = {
                k: torch.as_tensor(v, device="cuda")
                for k, v in minibatch_dict.items()
                if v is not None
            }

            return iter([(minibatch_dict, current_batches[0], current_batches[-1])])
        else:
            self.__writer.write_minibatches(minibatch_dict)
            return None

    def __get_call_groups(
        self,
        seeds: TensorType,
        input_id: TensorType,
        seeds_per_call: int,
        assume_equal_input_size: bool = False,
    ):
        # Split the input seeds into call groups.  Each call group
        # corresponds to one sampling call.  A call group contains
        # many batches.
        seeds_call_groups = torch.split(seeds, seeds_per_call, dim=-1)
        index_call_groups = torch.split(input_id, seeds_per_call, dim=-1)

        # Need to add empties to the list of call groups to handle the case
        # where not all ranks have the same number of call groups.  This
        # prevents a hang since we need all ranks to make the same number
        # of calls.
        if not assume_equal_input_size:
            num_call_groups = torch.tensor(
                [len(seeds_call_groups)], device="cuda", dtype=torch.int32
            )
            torch.distributed.all_reduce(
                num_call_groups, op=torch.distributed.ReduceOp.MAX
            )
            seeds_call_groups = list(seeds_call_groups) + (
                [torch.tensor([], dtype=seeds.dtype, device="cuda")]
                * (int(num_call_groups) - len(seeds_call_groups))
            )
            index_call_groups = list(index_call_groups) + (
                [torch.tensor([], dtype=torch.int64, device=index_call_groups.device)]
                * (int(num_call_groups) - len(index_call_groups))
            )

        return seeds_call_groups, index_call_groups

    def sample_from_nodes(
        self,
        nodes: TensorType,
        *,
        batch_size: int = 16,
        random_state: int = 62,
        assume_equal_input_size: bool = False,
        input_id: Optional[TensorType] = None,
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
        assume_equal_input_size: bool
            Whether the inputs across workers should be assumed to be equal in
            dimension.  Skips some checks if True.
        input_id: Optional[TensorType]
            Input ids corresponding to the original batch tensor, if it
            was permuted prior to calling this function.  If present,
            will be saved with the samples.
        """
        torch = import_optional("torch")

        nodes = torch.as_tensor(nodes, device="cuda")
        num_seeds = nodes.numel()

        batches_per_call = self._local_seeds_per_call // batch_size
        actual_seeds_per_call = batches_per_call * batch_size

        if input_id is None:
            input_id = torch.arange(num_seeds, dtype=torch.int64, device="cpu")

        local_num_batches = int(ceil(num_seeds / batch_size))
        batch_id_start, input_size_is_equal = self.get_start_batch_offset(
            local_num_batches, assume_equal_input_size=assume_equal_input_size
        )

        nodes_call_groups, index_call_groups = self.__get_call_groups(
            nodes,
            input_id,
            actual_seeds_per_call,
            assume_equal_input_size=input_size_is_equal,
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
                zip(nodes_call_groups, index_call_groups),
                self.__sample_from_nodes_func,
                *sample_args,
            )
        else:
            # Unbuffered sampling
            for i, current_seeds_and_ix in enumerate(
                zip(nodes_call_groups, index_call_groups)
            ):
                self.__sample_from_nodes_func(
                    i,
                    current_seeds_and_ix,
                    *sample_args,
                )

            # Return a reader that points to the stored samples
            rank = torch.distributed.get_rank() if self.is_multi_gpu else None
            return self.__writer.get_reader(rank)

    def __sample_from_edges_func(
        self,
        call_id: int,
        current_seeds_and_ix: Tuple["torch.Tensor", "torch.Tensor"],
        batch_id_start: int,
        batch_size: int,
        batches_per_call: int,
        random_state: int,
        assume_equal_input_size: bool,
    ) -> Union[None, Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]]:
        torch = import_optional("torch")

        current_seeds, current_ix = current_seeds_and_ix
        num_seed_edges = current_ix.numel()

        # The index gets stored as-is regardless of what makes it into
        # the final batch and in what order.
        # do qr division to get the number of batch_size batches and the
        # size of the last batch
        num_whole_batches, last_count = divmod(num_seed_edges, batch_size)
        input_offsets = torch.concatenate(
            [
                torch.tensor([0], device="cuda", dtype=torch.int64),
                torch.full(
                    (num_whole_batches,), batch_size, device="cuda", dtype=torch.int64
                ),
                torch.tensor([last_count], device="cuda", dtype=torch.int64)
                if last_count > 0
                else torch.tensor([], device="cuda", dtype=torch.int64),
            ]
        ).cumsum(-1)

        current_seeds, leftover_seeds = (
            current_seeds[:, num_whole_batches],
            current_seeds[:, num_whole_batches:],
        )

        # For input edges, we need to translate this into unique vertices
        # for each batch.
        # We start by reorganizing the seed and index tensors so we can
        # determine the unique vertices.  This results in the expected
        # src-to-dst concatenation for each batch
        current_seeds = torch.concat(
            [
                current_seeds[0].reshape((-1, batch_size)),
                current_seeds[1].reshape((-1, batch_size)),
            ],
            axis=-1,
        )

        # The returned unique values must be sorted or else the inverse won't line up
        # In the future this may be a good target for a C++ function
        # Each element is a tuple of (unique, index, inverse)
        # The seeds must be presorted with a stable sort prior to calling
        # unique_consecutive in order to support negative sampling.  This is
        # because if we put positive edges after negative ones, then we may
        # inadvertently turn a true positive into a false negative.
        y = (
            torch.sort(
                t,
                stable=True,
            )
            for t in current_seeds
        )
        z = ((v, torch.sort(i)[1]) for v, i in y)

        u = [
            (
                torch.unique_consecutive(
                    t,
                    return_inverse=True,
                ),
                i,
            )
            for t, i in z
        ]
        current_seeds = torch.concat([a[0] for a, _ in u])
        current_inv = torch.concat([a[1][i] for a, i in u])
        current_batches = torch.concat(
            [
                torch.full(
                    (a[0].numel(),),
                    i + batch_id_start + (call_id * batches_per_call),
                    device="cuda",
                    dtype=torch.int32,
                )
                for i, a in enumerate(u)
            ]
        )
        del u

        # Join with the leftovers
        leftover_seeds, lyi = torch.sort(
            leftover_seeds.flatten(),
            stable=True,
        )
        lz = torch.sort(lyi)[1]
        leftover_seeds, lui = leftover_seeds.unique_consecutive(return_inverse=True)
        leftover_inv = lui[lz]

        current_seeds = torch.concat([current_seeds, leftover_seeds])
        current_inv = torch.concat([current_inv, leftover_inv])
        current_batches = torch.concat(
            [
                current_batches,
                torch.full(
                    (leftover_seeds.numel(),),
                    current_batches[-1] + 1,
                    device="cuda",
                    dtype=torch.int32,
                ),
            ]
        )
        del leftover_seeds
        del lz
        del lui

        minibatch_dict = self.sample_batches(
            seeds=current_seeds,
            batch_ids=current_batches,
            random_state=random_state,
            assume_equal_input_size=assume_equal_input_size,
        )
        minibatch_dict["input_index"] = current_ix.cuda()
        minibatch_dict["input_offsets"] = input_offsets
        minibatch_dict[
            "edge_inverse"
        ] = current_inv  # (2 * batch_size) entries per batch

        if self.__writer is None:
            # rename renumber_map -> map to match unbuffered format
            minibatch_dict["map"] = minibatch_dict["renumber_map"]
            del minibatch_dict["renumber_map"]
            minibatch_dict = {
                k: torch.as_tensor(v, device="cuda")
                for k, v in minibatch_dict.items()
                if v is not None
            }

            return iter([(minibatch_dict, current_batches[0], current_batches[-1])])
        else:
            self.__writer.write_minibatches(minibatch_dict)
            return None

    def sample_from_edges(
        self,
        edges: TensorType,
        *,
        batch_size: int = 16,
        random_state: int = 62,
        assume_equal_input_size: bool = False,
        input_id: Optional[TensorType] = None,
    ) -> Iterator[Tuple[Dict[str, "torch.Tensor"], int, int]]:
        """
        Performs sampling starting from seed edges.

        Parameters
        ----------
        edges: TensorType
            2 x (# edges) tensor of edges to sample from.
            Standard src/dst format.  This will be converted
            to a list of seed nodes.
        batch_size: int
            The size of each batch.
        random_state: int
            The random seed to use for sampling.
        assume_equal_input_size: bool
            Whether this function should assume that inputs
            are equal across ranks.  Skips some potentially
            slow steps if True.
        input_id: Optional[TensorType]
            Input ids corresponding to the original batch tensor, if it
            was permuted prior to calling this function.  If present,
            will be saved with the samples.
        """

        torch = import_optional("torch")

        edges = torch.as_tensor(edges, device="cuda")
        num_seed_edges = edges.shape[-1]

        batches_per_call = self._local_seeds_per_call // batch_size
        actual_seed_edges_per_call = batches_per_call * batch_size

        if input_id is None:
            input_id = torch.arange(len(edges), dtype=torch.int64, device="cpu")

        local_num_batches = int(ceil(num_seed_edges / batch_size))
        batch_id_start, input_size_is_equal = self.get_start_batch_offset(
            local_num_batches, assume_equal_input_size=assume_equal_input_size
        )

        edges_call_groups, index_call_groups = self.__get_call_groups(
            edges,
            input_id,
            actual_seed_edges_per_call,
            assume_equal_input_size=input_size_is_equal,
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
                zip(edges_call_groups, index_call_groups),
                self.__sample_from_edges_func,
                *sample_args,
            )
        else:
            # Unbuffered sampling
            for i, current_seeds_and_ix in enumerate(
                zip(edges_call_groups, index_call_groups)
            ):
                self.__sample_from_edges_func(
                    i,
                    current_seeds_and_ix,
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
