# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import logging
import torch
import torch.distributed as td
import torch.nn.functional as F
from torchmetrics import Accuracy
from trainers import Trainer
import time


def get_features(input_nodes, output_nodes, feature_store, key="paper"):
    if isinstance(input_nodes, dict):
        input_nodes = input_nodes[key]
    if isinstance(output_nodes, dict):
        output_nodes = output_nodes[key]

    # TODO: Fix below
    # Adding based on assumption that cpu features
    # and gpu index is not supported yet

    if feature_store.backend == "torch":
        input_nodes = input_nodes.to("cpu")
        output_nodes = output_nodes.to("cpu")

    x = feature_store.get_data(indices=input_nodes, type_name=key, feat_name="x")
    y = feature_store.get_data(indices=output_nodes, type_name=key, feat_name="y")
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
    rank,
):
    time_forward_iter = time_forward / num_batches
    time_backward_iter = time_backward / num_batches
    total_time_iter = (time.perf_counter() - time_start) / num_batches
    logger.info(f"epoch {epoch}, iteration {iter_i}, rank {rank}")
    logger.info(f"time forward: {time_forward_iter}")
    logger.info(f"time backward: {time_backward_iter}")
    logger.info(f"loader time: {loader_time_iter}")
    logger.info(f"total time: {total_time_iter}")


def train_epoch(
    model, optimizer, loader, feature_store, epoch, num_classes, time_d, logger, rank
):
    """
    Train the model for one epoch.
        model: The model to train.
        optimizer: The optimizer to use.
        loader: The loader to use.
        data: cuGraph.gnn.FeatueStore
        epoch: The epoch number.
        num_classes: The number of classes.
        time_d: A dictionary of times.
        logger: The logger to use.
        rank: Total rank
    """
    model = model.train()
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
        x, y_true = get_features(input_nodes, output_nodes, feature_store=feature_store)
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

        if iter_i % 50 == 0:
            log_batch(
                logger=logger,
                iter_i=iter_i,
                num_batches=num_batches,
                time_forward=time_forward,
                time_backward=time_backward,
                time_start=time_start,
                loader_time_iter=loader_time_iter,
                epoch=epoch,
                rank=rank,
            )

    time_d["time_loader"] += time_loader
    time_d["time_feature_indexing"] += time_feature_indexing
    time_d["time_feature_transfer"] += time_feature_transfer
    time_d["time_forward"] += time_forward
    time_d["time_backward"] += time_backward

    return num_batches, total_loss


def get_accuracy(model, loader, feature_store, num_classes):
    print("Computing accuracy...", flush=True)
    acc = Accuracy(task="multiclass", num_classes=num_classes).cuda()
    acc_sum = 0.0
    num_batches = 0
    with torch.no_grad():
        for iter_i, (input_nodes, output_nodes, blocks) in enumerate(loader):
            x, y_true = get_features(
                input_nodes, output_nodes, feature_store=feature_store
            )
            x = x.to("cuda")
            y_true = y_true.to("cuda")

            out = model(blocks, x)
            batch_size = out.shape[0]
            acc_sum += acc(out[:batch_size].softmax(dim=-1), y_true[:batch_size])
            num_batches += 1

    num_batches = num_batches

    acc_sum = torch.tensor(float(acc_sum), dtype=torch.float32, device="cuda")
    td.all_reduce(acc_sum, op=td.ReduceOp.SUM)
    nb = torch.tensor(float(num_batches), dtype=torch.float32, device=acc_sum.device)
    td.all_reduce(nb, op=td.ReduceOp.SUM)

    acc = acc_sum / nb

    print(
        f"Accuracy: {acc * 100.0:.4f}%",
    )
    return acc * 100.0


class DGLTrainer(Trainer):
    def train(self):
        logger = logging.getLogger("DGLTrainer")
        time_d = {
            "time_loader": 0.0,
            "time_feature_indexing": 0.0,
            "time_feature_transfer": 0.0,
            "time_forward": 0.0,
            "time_backward": 0.0,
        }
        total_batches = 0
        for epoch in range(self.num_epochs):
            start_time = time.perf_counter()
            self.model.train()
            with td.algorithms.join.Join(
                [self.model], divide_by_initial_world_size=False
            ):
                num_batches, total_loss = train_epoch(
                    model=self.model,
                    optimizer=self.optimizer,
                    loader=self.get_loader(epoch=epoch, stage="train"),
                    feature_store=self.data,
                    num_classes=self.dataset.num_labels,
                    epoch=epoch,
                    time_d=time_d,
                    logger=logger,
                    rank=self.rank,
                )
                total_batches = total_batches + num_batches
            end_time = time.perf_counter()
            epoch_time_taken = end_time - start_time
            print(
                f"RANK: {self.rank} Total time taken for training epoch {epoch} = {epoch_time_taken}",
                flush=True,
            )
            print("---" * 30)
            td.barrier()
            self.model.eval()
            with td.algorithms.join.Join(
                [self.model], divide_by_initial_world_size=False
            ):
                # test
                test_acc = get_accuracy(
                    model=self.model.module,
                    loader=self.get_loader(epoch=epoch, stage="test"),
                    feature_store=self.data,
                    num_classes=self.dataset.num_labels,
                )
                print(f"Accuracy: {test_acc:.4f}%")

        # val:
        self.model.eval()
        with td.algorithms.join.Join([self.model], divide_by_initial_world_size=False):
            val_acc = get_accuracy(
                model=self.model.module,
                loader=self.get_loader(epoch=epoch, stage="val"),
                feature_store=self.data,
                num_classes=self.dataset.num_labels,
            )
            print(f"Validation Accuracy: {val_acc:.4f}%")

        val_acc = float(val_acc)
        stats = {
            "Accuracy": val_acc,
            "# Batches": total_batches,
            "Loader Time": time_d["time_loader"],
            "Feature Time": time_d["time_feature_indexing"]
            + time_d["time_feature_transfer"],
            "Forward Time": time_d["time_forward"],
            "Backward Time": time_d["time_backward"],
        }
        return stats


# For native DGL training, see benchmarks/cugraph-dgl/scale-benchmarks
