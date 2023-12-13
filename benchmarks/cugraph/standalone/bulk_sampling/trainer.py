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

import torch
import torch.distributed as td
import torch.nn.functional as F

import time

from typing import Union, List

def extend_tensor(t: Union[List[int], torch.Tensor], l:int):
    t = torch.as_tensor(t)

    return torch.concat([
        t,
        torch.zeros(
            l - len(t),
            dtype=t.dtype,
            device=t.device
        )
    ])

class Trainer:
    @property
    def model(self):
        raise NotImplementedError()
    
    @property
    def dataset(self):
        raise NotImplementedError()

    @property
    def data(self):
        raise NotImplementedError()

    @property
    def optimizer(self):
        raise NotImplementedError()
    
    @property
    def num_epochs(self) -> int:
        raise NotImplementedError()
    
    def get_loader(self, epoch: int):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

class PyGTrainer(Trainer):
    def train(self):
        import logging
        logger = logging.getLogger('PyGTrainer')

        total_loss = 0.0
        num_batches = 0

        time_forward = 0.0
        time_backward = 0.0
        time_loader = 0.0
        time_feature_additional = 0.0
        start_time = time.perf_counter()
        end_time_backward = start_time

        for epoch in range(self.num_epochs):
            with td.algorithms.join.Join([self.model, self.optimizer]):
                for iter_i, data in enumerate(self.get_loader(epoch)):
                    loader_time_iter = time.perf_counter() - end_time_backward
                    time_loader += loader_time_iter

                    additional_feature_time_start = time.perf_counter()
                    
                    num_sampled_nodes = sum([
                        torch.tensor(n)
                        for n in data.num_sampled_nodes_dict.values()
                    ])
                    num_sampled_edges = sum([
                        torch.tensor(e)
                        for e in data.num_sampled_edges_dict.values()
                    ])

                    # FIXME find a way to get around this and not have to call extend_tensor
                    num_layers = len(self.model.module.convs)
                    num_sampled_nodes = extend_tensor(num_sampled_nodes, num_layers + 1)
                    num_sampled_edges = extend_tensor(num_sampled_edges, num_layers)

                    data = data.to_homogeneous().cuda()
                    additional_feature_time_end = time.perf_counter()
                    time_feature_additional += additional_feature_time_end - additional_feature_time_start

                    num_batches += 1
                    if iter_i % 20 == 1:
                        time_forward_iter = time_forward / num_batches
                        time_backward_iter = time_backward / num_batches
                        
                        total_time_iter = (time.perf_counter() - start_time) / num_batches
                        logger.info(f"iteration {iter_i}")
                        logger.info(f"num sampled nodes: {num_sampled_nodes}")
                        logger.info(f"num sampled edges: {num_sampled_edges}")
                        logger.info(f"time forward: {time_forward_iter}")
                        logger.info(f"time backward: {time_backward_iter}")
                        logger.info(f"loader time: {loader_time_iter}")
                        logger.info(f"total time: {total_time_iter}")

                    
                    y_true = data.y
                    x = data.x.to(torch.float32)

                    start_time_forward = time.perf_counter()
                    edge_index = data.edge_index if 'edge_index' in data else data.adj_t
                    
                    y_pred = self.model(
                        x,
                        edge_index,
                        num_sampled_nodes,
                        num_sampled_edges,
                    )
                    
                    end_time_forward = time.perf_counter()
                    time_forward += end_time_forward - start_time_forward
                    
                    if y_pred.shape[0] > len(y_true):
                        raise ValueError(f"illegal shape: {y_pred.shape}; {y_true.shape}")

                    y_true = y_true[:y_pred.shape[0]]

                    y_true = F.one_hot(
                        y_true.to(torch.int64), num_classes=self.dataset.num_labels
                    ).to(torch.float32)            

                    if y_true.shape != y_pred.shape:
                        raise ValueError(
                            f'y_true shape was {y_true.shape} '
                            f'but y_pred shape was {y_pred.shape} '
                            f'in iteration {iter_i} '
                            f'on rank {y_pred.device.index}'
                        )                  

                    start_time_backward = time.perf_counter()
                    loss = F.cross_entropy(y_pred, y_true)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                    end_time_backward = time.perf_counter()
                    time_backward += end_time_backward - start_time_backward
            
            end_time = time.perf_counter()
            # FIXME add test, validation steps
        
        stats = {
            'Loss': total_loss,
            '# Batches': num_batches,
            'Loader Time': time_loader + time_feature_additional,
            'Forward Time': time_forward,
            'Backward Time': time_backward,
        }
        return stats