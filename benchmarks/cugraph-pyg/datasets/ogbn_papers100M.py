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

from .dataset import Dataset
from typing import Dict, Tuple, Union

import pandas
import torch
import numpy as np

from sklearn.model_selection import train_test_split

import gc
import os

# TODO automatically generate this dataset and splits
class OGBNPapers100MDataset(Dataset):
    def __init__(self, *, replication_factor=1, dataset_dir='.', train_split=0.8, val_split=0.5, load_edge_index=True):
        self.__replication_factor = replication_factor
        self.__disk_x = None
        self.__y = None
        self.__edge_index = None
        self.__dataset_dir = dataset_dir
        self.__train_split = train_split
        self.__val_split = val_split
        self.__load_edge_index = load_edge_index

    @property 
    def edge_index_dict(self) -> Dict[Tuple[str, str, str], Union[Dict[str, torch.Tensor], int]]:
        import logging
        logger = logging.getLogger('OGBNPapers100MDataset')

        if self.__edge_index is None:
            if self.__load_edge_index:
                npy_path = os.path.join(
                    self.__dataset_dir,
                    'ogbn_papers100M_copy',
                    'npy',
                    'paper__cites__paper',
                    'edge_index.npy'
                )

                logger.info(f'loading edge index from {npy_path}')
                ei = np.load(npy_path, mmap_mode='r')
                ei = torch.as_tensor(ei)
                ei = {
                    'src': ei[1],
                    'dst': ei[0],
                }

                logger.info('sorting edge index...')
                ei['dst'], ix = torch.sort(ei['dst'])
                ei['src'] = ei['src'][ix]
                del ix
                gc.collect()

                logger.info('processing replications...')
                orig_num_nodes = self.num_nodes('paper') // self.__replication_factor
                if self.__replication_factor > 1:
                    orig_src = ei['src'].clone().detach()
                    orig_dst = ei['dst'].clone().detach()
                    for r in range(1, self.__replication_factor):
                        ei['src'] = torch.concat([
                            ei['src'],
                            orig_src + int(r * orig_num_nodes),
                        ])

                        ei['dst'] = torch.concat([
                            ei['dst'],
                            orig_dst + int(r * orig_num_nodes),
                        ])

                    del orig_src
                    del orig_dst

                    ei['src'] = ei['src'].contiguous()
                    ei['dst'] = ei['dst'].contiguous()
                gc.collect()

                logger.info(f"# edges: {len(ei['src'])}")
                self.__edge_index = {('paper','cites','paper'): ei}
            else:
                self.__edge_index = {('paper','cites','paper'): self.num_edges(('paper','cites','paper'))}

        return self.__edge_index

    @property
    def x_dict(self) -> Dict[str, torch.Tensor]:
        node_type_path = os.path.join(
            self.__dataset_dir,
            'ogbn_papers100M',
            'npy',
            'paper'
        )
        
        if self.__disk_x is None:
            if self.__replication_factor == 1:
                full_path = os.path.join(node_type_path, 'node_feat.npy')
            else:
                full_path = os.path.join(node_type_path, f'node_feat_{self.__replication_factor}x.npy')
            
            self.__disk_x = {'paper': np.load(
                full_path,
                mmap_mode='r'
            )}

        return self.__disk_x

    @property
    def y_dict(self) -> Dict[str, torch.Tensor]:
        if self.__y is None:
            self.__get_labels()
        
        return self.__y
    
    @property
    def train_dict(self) -> Dict[str, torch.Tensor]:
        if self.__train is None:
            self.__get_labels()
        return self.__train
    
    @property
    def test_dict(self) -> Dict[str, torch.Tensor]:
        if self.__test is None:
            self.__get_labels()
        return self.__test
    
    @property
    def val_dict(self) -> Dict[str, torch.Tensor]:
        if self.__val is None:
            self.__get_labels()
        return self.__val
    
    @property
    def num_input_features(self) -> int:
        return int(self.x_dict['paper'].shape[1])
    
    @property
    def num_labels(self) -> int:
        return int(self.y_dict['paper'].max()) + 1
    
    def num_nodes(self, node_type: str) -> int:
        if node_type != 'paper':
            raise ValueError(f"Invalid node type {node_type}")

        return 111_059_956 * self.__replication_factor
    
    def num_edges(self, edge_type: Tuple[str, str, str]) -> int:
        if edge_type != ('paper', 'cites', 'paper'):
            raise ValueError(f"Invalid edge type {edge_type}")

        return 1_615_685_872 * self.__replication_factor
    
    def __get_labels(self):
        label_path = os.path.join(
            self.__dataset_dir,
            'ogbn_papers100M',
            'parquet',
            'paper',
            'node_label.parquet'
        )

        node_label = pandas.read_parquet(label_path)

        if self.__replication_factor > 1:
            orig_num_nodes = self.num_nodes('paper') // self.__replication_factor
            dfr = pandas.DataFrame({
                'node': pandas.concat([node_label.node + (r * orig_num_nodes) for r in range(1, self.__replication_factor)]),
                'label': pandas.concat([node_label.label for r in range(1, self.__replication_factor)]),
            })
            node_label = pandas.concat([node_label, dfr]).reset_index(drop=True)

        num_nodes = self.num_nodes("paper")
        node_label_tensor = torch.full((num_nodes,), -1, dtype=torch.float32, device='cpu')
        node_label_tensor[torch.as_tensor(node_label.node.values, device='cpu')] = \
            torch.as_tensor(node_label.label.values, device='cpu')
            
        self.__y = {'paper': node_label_tensor.contiguous()}
        
        train_ix, test_val_ix = train_test_split(torch.as_tensor(node_label.node.values), train_size=self.__train_split, random_state=num_nodes)
        test_ix, val_ix = train_test_split(test_val_ix, test_size=self.__val_split, random_state=num_nodes)

        train_tensor = torch.full((num_nodes,), 0, dtype=torch.bool, device='cpu')
        train_tensor[train_ix] = 1
        self.__train = {'paper': train_tensor}

        test_tensor = torch.full((num_nodes,), 0, dtype=torch.bool, device='cpu')
        test_tensor[test_ix] = 1
        self.__test = {'paper': test_tensor}

        val_tensor = torch.full((num_nodes,), 0, dtype=torch.bool, device='cpu')
        val_tensor[val_ix] = 1
        self.__val = {'paper': val_tensor}