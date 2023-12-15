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


class Trainer:
    @property
    def rank(self):
        raise NotImplementedError()

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

    def get_loader(self, epoch: int = 0, stage="train"):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()


def get_features(input_nodes, output_nodes, X, Y):
    x = X[input_nodes.to(X.device)]
    if isinstance(output_nodes, dict):
        output_nodes = output_nodes["paper"]

    if isinstance(Y, dict):
        Y = Y["paper"]["y"]

    output_nodes = output_nodes.to(Y.device)
    y = Y[output_nodes]
    return x, y


def log_batch(
    logger,
    iter_i,
    num_batches,
    time_forward,
    time_backward,
    time_start,
    loader_time_iter,
    epoch,
):
    time_forward_iter = time_forward / num_batches
    time_backward_iter = time_backward / num_batches
    total_time_iter = (time.perf_counter() - time_start) / num_batches
    logger.info(f"epoch {epoch}, iteration {iter_i}")
    logger.info(f"time forward: {time_forward_iter}")
    logger.info(f"time backward: {time_backward_iter}")
    logger.info(f"loader time: {loader_time_iter}")
    logger.info(f"total time: {total_time_iter}")


def train_epoch(model, optimizer, loader, X, Y, epoch, num_classes, time_d, logger):
    time_feature_indexing = time_d["time_feature_indexing"]
    time_feature_transfer = time_d["time_feature_transfer"]
    time_forward = time_d["time_forward"]
    time_backward = time_d["time_backward"]
    time_loader = time_d["time_loader"]

    time_start = time.perf_counter()
    end_time_backward = time.perf_counter()

    num_batches = 0
    total_loss = 0.0

    for iter_i, (input_nodes, output_nodes, blocks) in enumerate(loader):
        loader_time_iter = time.perf_counter() - end_time_backward
        time_loader += loader_time_iter
        feature_indexing_time_start = time.perf_counter()
        x, y_true = get_features(input_nodes, output_nodes, X, Y)
        additional_feature_time_end = time.perf_counter()
        time_feature_indexing += (
            additional_feature_time_end - feature_indexing_time_start
        )
        feature_trasfer_time_start = time.perf_counter()
        x = x.to("cuda")
        y_true = y_true.to("cuda")
        time_feature_transfer += time.perf_counter() - feature_trasfer_time_start
        num_batches += 1

        start_time_forward = time.perf_counter()
        y_pred = model(
            blocks,
            x,
        )
        end_time_forward = time.perf_counter()
        time_forward += end_time_forward - start_time_forward

        if y_pred.shape[0] > len(y_true):
            raise ValueError(f"illegal shape: {y_pred.shape}; {y_true.shape}")

        y_true = y_true[: y_pred.shape[0]]
        y_true = F.one_hot(
            y_true.to(torch.int64),
            num_classes=num_classes,
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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        end_time_backward = time.perf_counter()
        time_backward += end_time_backward - start_time_backward

        if iter_i % 20 == 0:
            log_batch(
                logger=logger,
                iter_i=iter_i,
                num_batches=num_batches,
                time_forward=time_forward,
                time_backward=time_backward,
                time_start=time_start,
                loader_time_iter=loader_time_iter,
                epoch=epoch,
            )

    time_d["time_loader"] = time_loader
    time_d["time_feature_indexing"] = time_feature_indexing
    time_d["time_feature_transfer"] = time_feature_transfer
    time_d["time_forward"] = time_forward
    time_d["time_backward"] = time_backward


class DGLTrainer(Trainer):
    def train(self):
        import logging

        logger = logging.getLogger("DGLTrainer")
        time_d = {
            "time_loader": 0.0,
            "time_feature_indexing": 0.0,
            "time_feature_transfer": 0.0,
            "time_forward": 0.0,
            "time_backward": 0.0,
        }
        for epoch in range(self.num_epochs):
            with td.algorithms.join.Join([self.model, self.optimizer]):
                train_epoch(
                    model=self.model,
                    optimizer=self.optimizer,
                    loader=self.get_loader(epoch=epoch, stage="train"),
                    X=self.data.x,
                    Y=self.data.y,
                    num_classes=self.dataset.num_labels,
                    epoch=epoch,
                    time_d=time_d,
                    logger=logger,
                )

            end_time = time.perf_counter()
            td.barrier()

        # test
        # from torchmetrics import Accuracy

        # acc = Accuracy(
        #     task="multiclass", num_classes=self.dataset.num_labels
        # ).cuda()

        # with td.algorithms.join.Join([self.model, self.optimizer]):
        #     if self.rank == 0:
        #         acc_sum = 0.0
        #         with torch.no_grad():
        #             for i, batch in enumerate(
        #                 self.get_loader(epoch=epoch, stage="test")
        #             ):
        #                 num_sampled_nodes = sum(
        #                     [
        #                         torch.tensor(n)
        #                         for n in batch.num_sampled_nodes_dict.values()
        #                     ]
        #                 )
        #                 num_sampled_edges = sum(
        #                     [
        #                         torch.tensor(e)
        #                         for e in batch.num_sampled_edges_dict.values()
        #                     ]
        #                 )
        #                 batch_size = num_sampled_nodes[0]

        #                 batch = batch.to_homogeneous().cuda()

        #                 batch.y = batch.y.to(torch.long)
        #                 out = self.model.module(
        #                     batch.x,
        #                     batch.edge_index,
        #                     num_sampled_nodes,
        #                     num_sampled_edges,
        #                 )
        #                 acc_sum += acc(
        #                     out[:batch_size].softmax(dim=-1), batch.y[:batch_size]
        #                 )
        #         print(
        #             f"Accuracy: {acc_sum/(i) * 100.0:.4f}%",
        #         )

        # with td.algorithms.join.Join([self.model, self.optimizer]):
        #     if self.rank == 0:
        #         acc_sum = 0.0
        #         with torch.no_grad():
        #             for i, batch in enumerate(
        #                 self.get_loader(epoch=epoch, stage="val")
        #             ):
        #                 num_sampled_nodes = sum(
        #                     [
        #                         torch.tensor(n)
        #                         for n in batch.num_sampled_nodes_dict.values()
        #                     ]
        #                 )
        #                 num_sampled_edges = sum(
        #                     [
        #                         torch.tensor(e)
        #                         for e in batch.num_sampled_edges_dict.values()
        #                     ]
        #                 )
        #                 batch_size = num_sampled_nodes[0]

        #                 batch = batch.to_homogeneous().cuda()

        #                 batch.y = batch.y.to(torch.long)
        #                 out = self.model.module(
        #                     batch.x,
        #                     batch.edge_index,
        #                     num_sampled_nodes,
        #                     num_sampled_edges,
        #                 )
        #                 acc_sum += acc(
        #                     out[:batch_size].softmax(dim=-1), batch.y[:batch_size]
        #                 )
        #         print(
        #             f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%",
        #         )

        # stats = {
        #     "Accuracy": (acc_sum / (i) * 100.0) if self.rank == 0 else 0.0,
        #     "# Batches": num_batches,
        #     "Loader Time": time_loader + time_feature_additional,
        #     "Forward Time": time_forward,
        #     "Backward Time": time_backward,
        # }
        return stats
