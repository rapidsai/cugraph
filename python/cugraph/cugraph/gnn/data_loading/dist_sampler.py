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

from typing import Union, List, Dict
TensorType = Union['torch.Tensor', 'cupy.ndarray', 'cudf.Series']


class DistSampleWriter:
    def __init__(self, format, directory, batches_per_partition):
        self.__format = format
        self.__directory = directory
        self.__batches_per_partition = batches_per_partition
    
    def write_minibatches(self, minibatch_dict):
        if ("majors" in minibatch_dict) and ("minors" in minibatch_dict):
            self.__write_minibatches_coo(minibatch_dict)
        elif "major_offsets" in minibatch_dict and "minors" in minibatch_dict:
            self.__write_minibatches_csr(minibatch_dict)
        else:
            raise ValueError("invalid columns")

class DistSampler:
    def __init__(self, graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph], writer: DistSampleWriter):
        self.__graph = graph
    
    def sample_batches(self, nodes: TensorType, batch: TensorType, random_state: int):
        raise NotImplementedError("Must be implemented by subclass")

    def sample_from_nodes(self, nodes: TensorType, batch: TensorType, random_state: int):
        minibatch_dict = self.sample_batches(nodes=nodes, batch=batch, random_state=random_state)
        self.__writer.write_minibatches(minibatch_dict)


    @property
    def is_multi_gpu(self):
        return isinstance(self.__graph, pylibcugraph.MGGraph)

class UniformNeighborSampler(DistSampler):
    def __init__(
            self,
            graph: Union[pylibcugraph.SGGraph, pylibcugraph.MGGraph],
            fanout: List[int],
            prior_sources_behavior: str,
            deduplicate_sources: bool,
            return_hops: bool,
            renumber: bool,
            compression: str,
            compress_per_hop:bool):
        super(graph)
        self.__fanout = fanout
        self.__prior_sources_behavior = prior_sources_behavior
        self.__deduplicate_sources = deduplicate_sources
        self.__return_hops = return_hops
        self.__renumber = renumber
        self.__compress_per_hop = compress_per_hop
        self.__compression = compression

    def sample_batches(self, nodes: TensorType, batch: TensorType, random_state: int):
        if self.is_multi_gpu:
            # get resource handle
            # pass label list, label_to_comm_rank
            # all labels on this worker should remain there
            pylibcugraph.uniform_neighbor_sample(

            )
        else:
            sampling_results_dict = pylibcugraph.uniform_neighbor_sample(
                pylibcugraph.ResourceHandle(),
                self.__graph,
                start_list=nodes,
                batch_id_list=batch,
                h_fan_out=np.array(self.__fanout, dtype='int32'),
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