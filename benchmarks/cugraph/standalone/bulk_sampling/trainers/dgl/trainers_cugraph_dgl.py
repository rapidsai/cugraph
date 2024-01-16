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
import os
import time

os.environ["LIBCUDF_CUFILE_POLICY"] = "KVIKIO"
os.environ["KVIKIO_NTHREADS"] = "32"
os.environ["RAPIDS_NO_INITIALIZE"] = "1"

from .trainers_dgl import DGLTrainer
from models.dgl import GraphSAGE

import torch
import numpy as np
import warnings

from torch.nn.parallel import DistributedDataParallel as ddp
from cugraph_dgl.dataloading import HomogenousBulkSamplerDataset
from cugraph.gnn import FeatureStore


def get_dataloader(input_file_paths, total_num_nodes, sparse_format, return_type):
    print("Creating dataloader", flush=True)
    st = time.time()
    dataset = HomogenousBulkSamplerDataset(
        total_num_nodes,
        edge_dir="in",
        sparse_format=sparse_format,
        return_type=return_type,
    )
    dataset.set_input_files(input_file_paths=input_file_paths)
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=lambda x: x, shuffle=False, num_workers=0, batch_size=None
    )
    et = time.time()
    print(f"Time to create dataloader = {et - st:.2f} seconds", flush=True)
    return dataloader


class DGLCuGraphTrainer(DGLTrainer):
    def __init__(
        self,
        dataset,
        model="GraphSAGE",
        device=0,
        rank=0,
        world_size=1,
        num_epochs=1,
        sample_dir=".",
        **kwargs,
    ):
        self.__data = None
        self.__device = device
        self.__rank = rank
        self.__world_size = world_size
        self.__num_epochs = num_epochs
        self.__dataset = dataset
        self.__sample_dir = sample_dir
        self.__loader_kwargs = kwargs
        self.__model = self.get_model(model)
        self.__optimizer = None

    @property
    def rank(self):
        return self.__rank

    @property
    def model(self):
        return self.__model

    @property
    def dataset(self):
        return self.__dataset

    @property
    def optimizer(self):
        if self.__optimizer is None:
            self.__optimizer = torch.optim.Adam(
                self.model.parameters(), lr=0.01, weight_decay=0.0005
            )
        return self.__optimizer

    @property
    def num_epochs(self) -> int:
        return self.__num_epochs

    def get_loader(self, epoch: int = 0, stage="train") -> int:
        # TODO support online sampling
        if stage in ["val", "test"]:
            path = os.path.join(self.__sample_dir, stage, "samples")
        else:
            path = os.path.join(self.__sample_dir, f"epoch={epoch}", stage, "samples")

        dataloader = get_dataloader(
            input_file_paths=self.get_input_files(path).tolist(),
            total_num_nodes=None,
            sparse_format="csc",
            return_type="cugraph_dgl.nn.SparseGraph",
        )
        return dataloader

    @property
    def data(self):
        import logging

        logger = logging.getLogger("DGLCuGraphTrainer")
        logger.info("getting data")

        if self.__data is None:
            # FIXME wholegraph
            fs = FeatureStore(backend="torch")
            num_nodes_dict = {}

            for node_type, x in self.__dataset.x_dict.items():
                logger.debug(f"getting x for {node_type}")
                fs.add_data(x, node_type, "x")
                num_nodes_dict[node_type] = self.__dataset.num_nodes(node_type)

            for node_type, y in self.__dataset.y_dict.items():
                logger.debug(f"getting y for {node_type}")
                fs.add_data(y, node_type, "y")

            for node_type, train in self.__dataset.train_dict.items():
                logger.debug(f"getting train for {node_type}")
                fs.add_data(train, node_type, "train")

            for node_type, test in self.__dataset.test_dict.items():
                logger.debug(f"getting test for {node_type}")
                fs.add_data(test, node_type, "test")

            for node_type, val in self.__dataset.val_dict.items():
                logger.debug(f"getting val for {node_type}")
                fs.add_data(val, node_type, "val")

            # # TODO support online sampling if the edge index is provided
            # num_edges_dict = self.__dataset.edge_index_dict
            # if not isinstance(list(num_edges_dict.values())[0], int):
            #     num_edges_dict = {k: len(v) for k, v in num_edges_dict}

            self.__data = fs
        return self.__data

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
                    model_backend="cugraph_dgl",
                )
                .to(torch.float32)
                .to(self.__device)
            )
            # TODO: Fix for distributed models
            if torch.distributed.is_initialized():
                model = ddp(model, device_ids=[self.__device])
            else:
                warnings.warn("Distributed training is not available")
            print("done creating model")

        return model

    def get_input_files(self, path):
        file_list = [entry.path for entry in os.scandir(path)]
        return np.array_split(file_list, self.__world_size)[self.__rank]