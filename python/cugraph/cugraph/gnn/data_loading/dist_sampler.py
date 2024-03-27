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

import pylibcugraph
import numpy as np
import cupy
import cudf

from typing import Union, List
from cugraph.utilities import import_optional
from cugraph.gnn import cugraph_comms_get_raft_handle

from math import ceil

# PyTorch is NOT optional but this is required for container builds.
torch = import_optional("torch")

TensorType = Union[torch.Tensor, cupy.ndarray, cudf.Series]


class DistSampleWriter:
    def __init__(self, format: str, directory: str, batches_per_partition: int):
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
    
    def __write_minibatches_coo(self):
        

    def write_minibatches(self, minibatch_dict):
        if ("majors" in minibatch_dict) and ("minors" in minibatch_dict):
            self.__write_minibatches_coo(minibatch_dict)
        elif "major_offsets" in minibatch_dict and "minors" in minibatch_dict:
            self.__write_minibatches_csr(minibatch_dict)
        else:
            raise ValueError("invalid columns")


class DistSampler:
    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        writer: DistSampleWriter,
        local_seeds_per_call: int = 32768,
        rank: int = 0,
    ):
        self.__graph = graph
        self.__writer = writer
        self.__local_seeds_per_call = local_seeds_per_call
        self.__rank = rank

    def sample_batches(
        self, seeds: TensorType, batch_ids: TensorType, random_state: int = 0
    ):
        raise NotImplementedError("Must be implemented by subclass")

    def sample_from_nodes(self, nodes: TensorType, batch_size: int, random_state: int):
        batches_per_call = self._local_seeds_per_call // batch_size
        actual_seeds_per_call = batches_per_call * batch_size
        num_calls = int(ceil(len(nodes) / actual_seeds_per_call))

        nodes = torch.split(torch.as_tensor(nodes, device="cuda"), num_calls)

        for i, current_seeds in enumerate(nodes):
            current_batches = torch.arange(
                i * batches_per_call,
                (i + 1) * batches_per_call,
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
            )
            self.__writer.write_minibatches(minibatch_dict)

    @property
    def is_multi_gpu(self):
        return isinstance(self.__graph, pylibcugraph.MGGraph)

    @property
    def _local_seeds_per_call(self):
        return self.__local_seeds_per_call

    @property
    def rank(self):
        return self.__rank


class UniformNeighborSampler(DistSampler):
    def __init__(
        self,
        graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
        fanout: List[int],
        prior_sources_behavior: str,
        deduplicate_sources: bool,
        compression: str,
        compress_per_hop: bool,
    ):
        super(graph)
        self.__fanout = fanout
        self.__prior_sources_behavior = prior_sources_behavior
        self.__deduplicate_sources = deduplicate_sources
        self.__compress_per_hop = compress_per_hop
        self.__compression = compression

    def sample_batches(
        self, seeds: TensorType, batch_ids: TensorType, random_state: int = 0
    ):
        if self.is_multi_gpu:
            handle = pylibcugraph.ResourceHandle(
                cugraph_comms_get_raft_handle().getHandle()
            )
            label_to_output_comm_rank = torch.full(
                (len(seeds),), self.rank, dtype=torch.int32, device="cuda"
            )
            label_list = torch.unique(batch_ids)

            sampling_results_dict = pylibcugraph.uniform_neighbor_sample(
                handle,
                self.__graph,
                start_list=seeds,
                batch_id_list=batch_ids,
                label_list=label_list,
                label_to_output_comm_rank=label_to_output_comm_rank,
                h_fan_out=np.array(self.__fanout, dtype="int32"),
                with_replacement=self.__with_replacement,
                do_expensive_check=False,
                with_edge_properties=True,
                random_state=random_state,
                prior_sources_behavior=self.__prior_sources_behavior,
                deduplicate_sources=self.__deduplicate_sources,
                return_hops=True,
                renumber=self.__renumber,
                compression=self.__compression,
                compress_per_hop=self.__compress_per_hop,
                return_dict=True,
            )
        else:
            sampling_results_dict = pylibcugraph.uniform_neighbor_sample(
                pylibcugraph.ResourceHandle(),
                self.__graph,
                start_list=seeds,
                batch_id_list=batch_ids,
                h_fan_out=np.array(self.__fanout, dtype="int32"),
                with_replacement=self.__with_replacement,
                do_expensive_check=False,
                with_edge_properties=True,
                random_state=random_state,
                prior_sources_behavior=self.__prior_sources_behavior,
                deduplicate_sources=self.__deduplicate_sources,
                return_hops=self.__return_hops,
                renumber=self.__renumber,
                compression=self.__compression,
                compress_per_hop=self.__compress_per_hop,
                return_dict=True,
            )

        return sampling_results_dict
