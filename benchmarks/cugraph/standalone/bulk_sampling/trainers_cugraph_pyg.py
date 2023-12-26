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

from trainers_pyg import PyGTrainer
from models_cugraph_pyg import CuGraphSAGE

import torch
import numpy as np

from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as ddp

from cugraph.gnn import FeatureStore
from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.loader import BulkSampleLoader

import gc
import os


class PyGCuGraphTrainer(PyGTrainer):
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
        return ZeroRedundancyOptimizer(
            self.model.parameters(), torch.optim.Adam, lr=0.01
        )

    @property
    def num_epochs(self) -> int:
        return self.__num_epochs

    def get_loader(self, epoch: int = 0, stage="train") -> int:
        import logging
        logger = logging.getLogger("PyGCuGraphTrainer")

        logger.info(f"getting loader for epoch {epoch}, {stage} stage")

        # TODO support online sampling
        if stage == "val":
            path = os.path.join(self.__sample_dir, "val", "samples")
        else:
            path = os.path.join(self.__sample_dir, f"epoch={epoch}", stage, "samples")
        
        loader = BulkSampleLoader(
            self.data,
            self.data,
            None,  # FIXME get input nodes properly
            directory=path,
            input_files=self.get_input_files(path),
            **self.__loader_kwargs,
        )

        logger.info(f"got loader successfully on rank {self.rank}")
        return loader

    @property
    def data(self):
        import logging

        logger = logging.getLogger("PyGCuGraphTrainer")
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

            # TODO support online sampling if the edge index is provided
            num_edges_dict = self.__dataset.edge_index_dict
            if not isinstance(list(num_edges_dict.values())[0], int):
                num_edges_dict = {k: len(v) for k, v in num_edges_dict}

            self.__data = CuGraphStore(
                fs,
                num_edges_dict,
                num_nodes_dict,
            )

        logger.info(f"got data successfully on rank {self.rank}")

        return self.__data

    def get_model(self, name="GraphSAGE"):
        if name != "GraphSAGE":
            raise ValueError("only GraphSAGE is currently supported")

        num_input_features = self.__dataset.num_input_features
        num_output_features = self.__dataset.num_labels
        num_layers = len(self.__loader_kwargs["num_neighbors"])

        with torch.cuda.device(self.__device):
            model = (
                CuGraphSAGE(
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

    def get_input_files(self, path):
        file_list = np.array(os.listdir(path))

        return np.array_split(file_list, self.__world_size)[self.__rank]
