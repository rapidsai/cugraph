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


from trainer import PyGTrainer
from datasets import OGBNPapers100MDataset
from models_native import GraphSAGE

import torch
import numpy as np

from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as ddp

from torch_geometric.utils.sparse import index2ptr
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

import gc
import os

def pyg_num_workers(world_size):
    num_workers = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_workers = len(os.sched_getaffinity(0)) / (2 * world_size)
        except Exception:
            pass
    if num_workers is None:
        num_workers = os.cpu_count() / (2 * world_size)
    return int(num_workers)

class PyGNativeTrainer(PyGTrainer):
    def __init__(self, dataset, model='GraphSAGE', device=0, rank=0, world_size=1, num_epochs=1, **kwargs):
        self.__dataset = dataset
        self.__device = device
        self.__data = None
        self.__rank = rank
        self.__num_epochs = num_epochs
        self.__world_size = world_size
        self.__loader_kwargs = kwargs
        self.__model = self.get_model(model)

    @property
    def model(self):
        return self.__model

    @property
    def dataset(self):
        return self.__dataset

    @property
    def data(self):
        import logging
        logger = logging.getLogger('PyGNativeTrainer')
        logger.info('getting data')
        
        if self.__data is None:
            self.__data = HeteroData()

            for node_type, x in self.__dataset.x_dict.items():
                logger.debug(f'getting x for {node_type}')
                self.__data[node_type].x = x
                self.__data[node_type]['num_nodes'] = self.__dataset.num_nodes(node_type)
            
            for node_type, y in self.__dataset.y_dict.items():
                logger.debug(f'getting y for {node_type}')
                self.__data[node_type]['y'] = y
            
            for node_type, train in self.__dataset.train_dict.items():
                logger.debug(f'getting train for {node_type}')
                self.__data[node_type]['train'] = train
            
            for node_type, test in self.__dataset.test_dict.items():
                logger.debug(f'getting test for {node_type}')
                self.__data[node_type]['test'] = test
            
            for node_type, val in self.__dataset.val_dict.items():
                logger.debug(f'getting val for {node_type}')
                self.__data[node_type]['val'] = val

            for can_edge_type, ei in self.__dataset.edge_index_dict.items():
                logger.info('converting to csc...')
                ei['dst'] = index2ptr(ei['dst'], self.__dataset.num_nodes(can_edge_type[2]))

                logger.info('updating data structure...')
                self.__data.put_edge_index(
                    layout='csc',
                    edge_index=list(ei.values()),
                    edge_type=can_edge_type,
                    size=(self.__dataset.num_nodes(can_edge_type[0]), self.__dataset.num_nodes(can_edge_type[2])),
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
        import logging
        logger = logging.getLogger('PyGNativeTrainer')
        logger.info(f'Getting loader for epoch {epoch}')

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
        loader = NeighborLoader(
            self.data,
            input_nodes=input_nodes,
            is_sorted=True,
            disjoint=False,
            num_workers=pyg_num_workers(self.__world_size), # FIXME change this
            persistent_workers=True,
            **self.__loader_kwargs # batch size, num neighbors, replace, shuffle, etc.
        )

        logger.info('done creating loader')
        return loader
    
    def get_model(self, name='GraphSAGE'):
        if name != 'GraphSAGE':
            raise ValueError("only GraphSAGE is currently supported")

        num_input_features = self.__dataset.num_input_features
        num_output_features = self.__dataset.num_labels
        num_layers = len(self.__loader_kwargs['num_neighbors'])

        with torch.cuda.device(self.__device):
            model = GraphSAGE(
                in_channels=num_input_features,
                hidden_channels=64,
                out_channels=num_output_features,
                num_layers=num_layers,
            ).to(torch.float32).to(self.__device)
            model = ddp(model, device_ids=[self.__device])
            print('done creating model')
        
        return model