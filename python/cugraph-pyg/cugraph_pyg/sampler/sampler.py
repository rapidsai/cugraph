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

from typing import Optional, Iterator, Union, Dict

from cugraph.utilities.utils import import_optional
from cugraph.gnn import DistSampler, DistSamplerReader

torch = import_optional("torch")
torch_geometric = import_optional('torch_geometric')

class SampleReader:
    def __init__(self, base_reader: DistSampleReader):
        self.__base_reader = base_reader
        self.__num_samples_remaining = 0
        self.__index = 0
    
    def __next__(self):
        if self.__num_samples_remaining == 0:
            # raw_sample_data is already a dict of tensors
            self.__raw_sample_data, start_inclusive, end_inclusive = next(self.__base_reader)

            self.__raw_sample_data['label_hop_offsets'] -= self.__raw_sample_data['label_hop_offsets'][0].clone()
            self.__raw_sample_data['renumber_map_offsets'] -= self.__raw_sample_data['renumber_map_offsets'][0].clone()
            if 'major_offsets' in self.__raw_sample_data:
                self.__raw_sample_data['major_offsets'] -= self.__raw_sample_data['major_offsets'][0].clone()

            self.__num_samples_remaining = end_inclusive - start_inclusive + 1
            self.__index = 0
            
        return self._decode(self.__raw_sample_data, self.__index)

    def __iter__(self):
        return self

class HomogeneousSampleReader(SampleReader):
    def __init__(self, base_reader: DistSampleReader):
        super().__init__(base_reader)

    def __decode_csc(self, raw_sample_data: Dict['torch.Tensor'], index: int):
        fanout_length = len(raw_sample_data['label_hop_offsets']) - 1 // (len(raw_sample_data['renumber_map_offsets']) - 1)
        
        major_offsets_start_incl = raw_sample_data['label_hop_offsets'][index * fanout_length]
        major_offsets_end_incl = raw_sample_data['label_hop_offsets'][(index + 1) * fanout_length]

        major_offsets = raw_sample_data['major_offsets'][major_offsets_start_incl : major_offsets_end_incl + 1].clone()
        minors = raw_sample_data['minors'][major_offsets[0] : major_offsets[-1]]
        edge_id = raw_sample_data['edge_id'][major_offsets[0] : major_offsets[-1]]
        # don't retrieve edge type for a homogeneous graph

        renumber_map_start = raw_sample_data['renumber_map_offsets'][index]
        renumber_map_end = raw_sample_data['renumber_map_offsets'][index + 1]

        renumber_map = raw_sample_data['map'][renumber_map_start:renumber_map_end]

        current_label_hop_offsets = raw_sample_data['label_hop_offsets'][index * fanout_length : (index + 1) * fanout_length + 1].clone()
        current_label_hop_offsets -= current_label_hop_offsets[0].clone()

        num_sampled_edges = major_offsets[current_label_hop_offsets].diff()
        num_sampled_nodes = torch.concat(
            [
                current_label_hop_offsets.diff(),
                (renumber_map.shape[0] - current_label_hop_offsets[-1]).reshape((1,)),
            ]
        )

        return torch_geometric.sampler.SamplerOutput(
            node=renumber_map,
            row=minors,
            col=major_offsets,
            edge=edge_id,
            batch=renumber_map[:num_sampled_nodes[0]],
            num_sampled_nodes=num_sampled_nodes.cpu(),
            num_sampled_edges=num_sampled_edges.cpu(),
        )

    def __decode_coo(raw_sample_data: Dict['torch.Tensor'], index: int):
        fanout_length = len(raw_sample_data['label_hop_offsets']) - 1 // (len(raw_sample_data['renumber_map_offsets']) - 1)
        
        major_minor_start = raw_sample_data['label_hop_offsets'][index * fanout_length]
        major_minor_end = raw_sample_data['label_hop_offsets'][(index + 1) * fanout_length]

        majors = raw_sample_data['majors'][major_minor_start:major_minor_end]
        minors = raw_sample_data['minors'][major_minor_start:major_minor_end]
        edge_id = raw_sample_data['edge_id'][major_minor_start:major_minor_end]
        # don't retrieve edge type for a homogeneous graph

        renumber_map_start = raw_sample_data['renumber_map_offsets'][index]
        renumber_map_end = raw_sample_data['renumber_map_offsets'][index + 1]

        renumber_map = raw_sample_data['map'][renumber_map_start:renumber_map_end]

        num_sampled_edges = raw_sample_data['label_hop_offsets'][index * fanout_length : (index + 1) * fanout_length + 1].diff().cpu()

        return torch_geometric.sampler.SamplerOutput(
            node=renumber_map,
            row=minors,
            col=majors,
            edge=edge_id,
            batch=None,
            num_sampled_nodes=None,
            num_sampled_edges=num_sampled_edges,
        )

    def _decode(self, raw_sample_data: Dict['torch.Tensor'], index: int):
        if 'major_offsets' in raw_sample_data:
            return self.__decode_csc(raw_sample_data, index)
        else:
            return self.__decode_coo(raw_sample_data, index)

class BaseSampler:
    def __init__(self, sampler: DistSampler):
        self.__sampler = sampler

    def sample_from_nodes(self, index: 'torch_geometric.sampler.NodeSamplerInput', **kwargs) -> Iterator[Union['torch_geometric.sampler.HeteroSamplerOutput', 'torch_geometric.sampler.SamplerOutput']]:
        self.__sampler.sample_from_nodes(
            index.node,
            **kwargs
        )

        return SampleReader(
            self.__sampler.get_reader()
        )

    def sample_from_edges(self, index: 'torch_geometric.sampler.EdgeSamplerInput', neg_sampling: Optional['torch_geometric.sampler.NegativeSampling'], **kwargs) -> Iterator[Union['torch_geometric.sampler.HeteroSamplerOutput', 'torch_geometric.sampler.SamplerOutput']]:
        raise NotImplementedError("Edge sampling is currently unimplemented.")