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
from trainer import extend_tensor
from datasets import OGBNPapers100MDataset
from models_pyg import GraphSAGE

import torch
import numpy as np

import torch.distributed as td
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as ddp
import torch.nn.functional as F

from torch_geometric.utils.sparse import index2ptr
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

import gc
import os
import time

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


class PyGTrainer(Trainer):
    def train(self):
        import logging

        logger = logging.getLogger("PyGTrainer")
        logger.info("Entered train loop")

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
                for iter_i, data in enumerate(
                    self.get_loader(epoch=epoch, stage="train")
                ):
                    logger.info(f"epoch {epoch}, iteration {iter_i}")

                    loader_time_iter = time.perf_counter() - end_time_backward
                    time_loader += loader_time_iter

                    additional_feature_time_start = time.perf_counter()

                    num_sampled_nodes = sum(
                        [torch.tensor(n) for n in data.num_sampled_nodes_dict.values()]
                    )
                    num_sampled_edges = sum(
                        [torch.tensor(e) for e in data.num_sampled_edges_dict.values()]
                    )

                    # FIXME find a way to get around this and not have to call extend_tensor
                    num_layers = len(self.model.module.convs)
                    num_sampled_nodes = extend_tensor(num_sampled_nodes, num_layers + 1)
                    num_sampled_edges = extend_tensor(num_sampled_edges, num_layers)

                    data = data.to_homogeneous().cuda()
                    additional_feature_time_end = time.perf_counter()
                    time_feature_additional += (
                        additional_feature_time_end - additional_feature_time_start
                    )

                    num_batches += 1
                    if iter_i % 20 == 1:
                        time_forward_iter = time_forward / num_batches
                        time_backward_iter = time_backward / num_batches

                        total_time_iter = (
                            time.perf_counter() - start_time
                        ) / num_batches
                        logger.info(f"epoch {epoch}, iteration {iter_i}")
                        logger.info(f"num sampled nodes: {num_sampled_nodes}")
                        logger.info(f"num sampled edges: {num_sampled_edges}")
                        logger.info(f"time forward: {time_forward_iter}")
                        logger.info(f"time backward: {time_backward_iter}")
                        logger.info(f"loader time: {loader_time_iter}")
                        logger.info(f"total time: {total_time_iter}")

                    y_true = data.y
                    x = data.x.to(torch.float32)

                    start_time_forward = time.perf_counter()
                    edge_index = data.edge_index if "edge_index" in data else data.adj_t

                    y_pred = self.model(
                        x,
                        edge_index,
                        num_sampled_nodes,
                        num_sampled_edges,
                    )

                    end_time_forward = time.perf_counter()
                    time_forward += end_time_forward - start_time_forward

                    if y_pred.shape[0] > len(y_true):
                        raise ValueError(
                            f"illegal shape: {y_pred.shape}; {y_true.shape}"
                        )

                    y_true = y_true[: y_pred.shape[0]]

                    y_true = F.one_hot(
                        y_true.to(torch.int64), num_classes=self.dataset.num_labels
                    ).to(torch.float32)

                    if y_true.shape != y_pred.shape:
                        raise ValueError(
                            f"y_true shape was {y_true.shape} "
                            f"but y_pred shape was {y_pred.shape} "
                            f"in iteration {iter_i} "
                            f"on rank {y_pred.device.index}"
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

            # test
            from torchmetrics import Accuracy

            acc = Accuracy(
                task="multiclass", num_classes=self.dataset.num_labels
            ).cuda()

            with td.algorithms.join.Join([self.model, self.optimizer]):
                if self.rank == 0:
                    acc_sum = 0.0
                    with torch.no_grad():
                        for i, batch in enumerate(
                            self.get_loader(epoch=epoch, stage="test")
                        ):
                            num_sampled_nodes = sum(
                                [
                                    torch.tensor(n)
                                    for n in batch.num_sampled_nodes_dict.values()
                                ]
                            )
                            num_sampled_edges = sum(
                                [
                                    torch.tensor(e)
                                    for e in batch.num_sampled_edges_dict.values()
                                ]
                            )
                            batch_size = num_sampled_nodes[0]

                            batch = batch.to_homogeneous().cuda()

                            batch.y = batch.y.to(torch.long)
                            out = self.model.module(
                                batch.x,
                                batch.edge_index,
                                num_sampled_nodes,
                                num_sampled_edges,
                            )
                            acc_sum += acc(
                                out[:batch_size].softmax(dim=-1), batch.y[:batch_size]
                            )
                    print(
                        f"Accuracy: {acc_sum/(i) * 100.0:.4f}%",
                    )

            td.barrier()

        with td.algorithms.join.Join([self.model, self.optimizer]):
            if self.rank == 0:
                acc_sum = 0.0
                with torch.no_grad():
                    for i, batch in enumerate(
                        self.get_loader(epoch=epoch, stage="val")
                    ):
                        num_sampled_nodes = sum(
                            [
                                torch.tensor(n)
                                for n in batch.num_sampled_nodes_dict.values()
                            ]
                        )
                        num_sampled_edges = sum(
                            [
                                torch.tensor(e)
                                for e in batch.num_sampled_edges_dict.values()
                            ]
                        )
                        batch_size = num_sampled_nodes[0]

                        batch = batch.to_homogeneous().cuda()

                        batch.y = batch.y.to(torch.long)
                        out = self.model.module(
                            batch.x,
                            batch.edge_index,
                            num_sampled_nodes,
                            num_sampled_edges,
                        )
                        acc_sum += acc(
                            out[:batch_size].softmax(dim=-1), batch.y[:batch_size]
                        )
                print(
                    f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%",
                )

        stats = {
            "Accuracy": float(acc_sum / (i) * 100.0) if self.rank == 0 else 0.0,
            "# Batches": num_batches,
            "Loader Time": time_loader + time_feature_additional,
            "Forward Time": time_forward,
            "Backward Time": time_backward,
        }
        return stats



class PyGNativeTrainer(PyGTrainer):
    def __init__(
        self,
        dataset,
        model="GraphSAGE",
        device=0,
        rank=0,
        world_size=1,
        num_epochs=1,
        **kwargs,
    ):
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

        logger = logging.getLogger("PyGNativeTrainer")
        logger.info("getting data")

        if self.__data is None:
            self.__data = HeteroData()

            for node_type, x in self.__dataset.x_dict.items():
                logger.debug(f"getting x for {node_type}")
                self.__data[node_type].x = x
                self.__data[node_type]["num_nodes"] = self.__dataset.num_nodes(
                    node_type
                )

            for node_type, y in self.__dataset.y_dict.items():
                logger.debug(f"getting y for {node_type}")
                self.__data[node_type]["y"] = y

            for node_type, train in self.__dataset.train_dict.items():
                logger.debug(f"getting train for {node_type}")
                self.__data[node_type]["train"] = train

            for node_type, test in self.__dataset.test_dict.items():
                logger.debug(f"getting test for {node_type}")
                self.__data[node_type]["test"] = test

            for node_type, val in self.__dataset.val_dict.items():
                logger.debug(f"getting val for {node_type}")
                self.__data[node_type]["val"] = val

            for can_edge_type, ei in self.__dataset.edge_index_dict.items():
                logger.info("converting to csc...")
                ei["dst"] = index2ptr(
                    ei["dst"], self.__dataset.num_nodes(can_edge_type[2])
                )

                logger.info("updating data structure...")
                self.__data.put_edge_index(
                    layout="csc",
                    edge_index=list(ei.values()),
                    edge_type=can_edge_type,
                    size=(
                        self.__dataset.num_nodes(can_edge_type[0]),
                        self.__dataset.num_nodes(can_edge_type[2]),
                    ),
                    is_sorted=True,
                )
                gc.collect()

        return self.__data

    @property
    def optimizer(self):
        return ZeroRedundancyOptimizer(
            self.model.parameters(), torch.optim.Adam, lr=0.01
        )

    @property
    def num_epochs(self) -> int:
        return self.__num_epochs

    def get_loader(self, epoch: int):
        import logging

        logger = logging.getLogger("PyGNativeTrainer")
        logger.info(f"Getting loader for epoch {epoch}")

        input_nodes_dict = {
            node_type: np.array_split(
                np.arange(len(train_mask))[train_mask], self.__world_size
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
            num_workers=pyg_num_workers(self.__world_size),  # FIXME change this
            persistent_workers=True,
            **self.__loader_kwargs,  # batch size, num neighbors, replace, shuffle, etc.
        )

        logger.info("done creating loader")
        return loader

    def get_model(self, name="GraphSAGE"):
        if name != "GraphSAGE":
            raise ValueError("only GraphSAGE is currently supported")

        num_input_features = self.__dataset.num_input_features
        num_output_features = self.__dataset.num_labels
        num_layers = len(self.__loader_kwargs["num_neighbors"])

        with torch.cuda.device(self.__device):
            model = (
                GraphSAGE(
                    in_channels=num_input_features,
                    hidden_channels=64,
                    out_channels=num_output_features,
                    num_layers=num_layers,
                )
                .to(torch.float32)
                .to(self.__device)
            )
            model = ddp(model, device_ids=[self.__device])
            print("done creating model")

        return model
