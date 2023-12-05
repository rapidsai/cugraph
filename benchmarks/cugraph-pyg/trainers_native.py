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


from trainer import Trainer
from datasets import OGBNPapers100MDataset
from models_native import GraphSAGE

import torch.distributed as td
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as ddp
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.data import HeteroData


class NativeTrainer(Trainer):
    def __init__(self, model='GraphSAGE', dataset, device, rank=0, world_size=1, num_epochs=1, **kwargs):
        self.__model = get_model(model)
        self.__dataset = dataset
        self.__device = device
        self.__data = None
        self.__rank = rank
        self.__num_epochs = num_epochs
        self.__world_size = world_size
        self.loader_kwargs = kwargs

    @property
    def model(self):
        return self.__model

    @property
    def data(self):
        if self.__data is None:
            self.__data = HeteroData()

            for node_type, x in self.__dataset.x_dict.items():
                self.__data[node_type].x = x
                self.__data[node_type]['num_nodes'] = self.__dataset.num_nodes(node_type)
            
            for node_type, y in self.__dataset.y_dict.items():
                self.__data[node_type]['y'] = y
            
            for node_type, train in self.__dataset.train_dict.items():
                self.__data[node_type]['train'] = train
            
            for node_type, test in self.__dataset.test_dict.items():
                self.__data[node_type]['test'] = test
            
            for node_type, val in self.__dataset.val_dict.items():
                self.__data[node_type]['val'] = val

            for can_edge_type, ei in self.__dataset.edge_index_dict.items():
                print('converting to csc...')
                ei['dst'] = index2ptr(ei['dst'], num_nodes_dict[can_edge_type[2]])

                print('updating data structure...')
                self.__data.put_edge_index(
                    layout='csc',
                    edge_index=list(ei.values()),
                    edge_type=can_edge_type,
                    size=(num_nodes_dict[can_edge_type[0]], num_nodes_dict[can_edge_type[2]]),
                    is_sorted=True
                )
                gc.collect()

        return self.__data
    
    @property
    def optimizer(self):
        return ZeroRedundancyOptimizer(self.model.parameters(), torch.optim.Adam, lr=0.01)
    
    @property
    def num_epochs(self) -> int:
        return self.__num_epochs
    
    def get_loader(self, epoch: int):
        input_nodes_dict = {
            node_type: np.array_split(
                np.arange(len(train_mask))[train_mask],
                self.__world_size
            )[self.__rank]
            for node_type, train_mask in self.__dataset.train_dict.items()
        }

        input_nodes = list(input_nodes_dict.items())
        if len(input_nodes) > 1:
            raise ValueError("Multiple input node types currently unsupported")
        else:
            input_nodes = tuple(input_nodes[0])

        # get loader
        return NeighborLoader(
            hetero_data,
            input_nodes=input_nodes,
            is_sorted=True,
            disjoint=False,
            num_workers=None, # FIXME change this
            persistent_workers=True,
            **kwargs # batch size, num neighbors, replace, shuffle, etc.
        )
    
    def get_model(self, name='GraphSAGE'):
        if name != 'GraphSAGE':
            raise ValueError("only GraphSAGE is currently supported")

        num_input_features = self.__dataset.num_input_features
        num_output_features = self.__dataset.num_labels

        with torch.cuda.device(self.__device):
            model = GraphSAGE(
                in_channels=num_input_features,
                hidden_channels=64,
                out_channels=num_output_features,
                num_layers=len(output_meta['fanout'])
            ).to(torch.float32).to(self.__device)
            model = ddp(model, device_ids=[self.__device])
            print('done creating model')