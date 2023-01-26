# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from __future__ import annotations
from typing import Tuple, Dict

import os
import cudf
from cugraph_dgl.dataloading.utils.sampling_helpers import (
    create_homogeneous_sampled_graphs_from_dataframe,
    create_heterogeneous_sampled_graphs_from_dataframe,
)

# TODO: Make optional imports
import torch
import dgl


# Todo: maybe should switch to __iter__
class HomogenousBulkSamplerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_batches: int,
        total_number_of_nodes: int,
        edge_dir: bool,
    ):
        self.num_batches = num_batches
        self.total_number_of_nodes = total_number_of_nodes
        self.edge_dir = edge_dir
        self._current_batch_fn = None
        self._input_directory = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self._input_directory is None:
            raise dgl.DGLError(
                "Please set input directory by calling `set_input_directory` "
                "before trying to fetch a sample"
            )

        fn, batch_offset = self._batch_to_fn_d[idx]
        if fn != self._current_batch_fn:
            df = cudf.read_parquet(os.path.join(self._input_directory, fn))
            if self.edge_dir == "in":
                df.rename(
                    columns={"sources": "destinations", "destinations": "sources"},
                    inplace=True,
                )
            self._current_batch_fn = fn
            self._current_batch_start = batch_offset
            self._current_batches = create_homogeneous_sampled_graphs_from_dataframe(
                df, self.total_number_of_nodes, self.edge_dir
            )

        current_offset = idx - batch_offset
        return self._current_batches[current_offset]

    def set_input_directory(self, input_directory):
        self._input_directory = input_directory
        self._sampled_files = os.listdir(input_directory)
        self._batch_to_fn_d = {
            i: get_batch_fn_batch_start(i, self._sampled_files)
            for i in range(0, self.num_batches + 1)
        }


# Todo: combine with above
class HetrogenousBulkSamplerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_batches: int,
        num_nodes_dict: Dict[str, int],
        etype_id_dict: Dict[int, Tuple[str, str, str]],
        etype_offset_dict: Dict[Tuple[str, str, str], int],
        ntype_offset_dict: Dict[str, int],
        edge_dir: str = "in",
    ):
        self.num_batches = num_batches
        self.num_nodes_dict = num_nodes_dict
        self.etype_id_dict = etype_id_dict
        self.etype_offset_dict = etype_offset_dict
        self.ntype_offset_dict = ntype_offset_dict
        self.edge_dir = edge_dir
        self._current_batch_fn = None
        self._input_directory = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self._input_directory is None:
            raise dgl.DGLError(
                "Please set input directory by calling `set_input_directory` "
                "before trying to fetch a sample"
            )

        fn, batch_offset = self._batch_to_fn_d[idx]
        if fn != self._current_batch_fn:
            df = cudf.read_parquet(os.path.join(self._input_directory, fn))
            if self.edge_dir == "in":
                df.rename(
                    columns={"sources": "destinations", "destinations": "sources"},
                    inplace=True,
                )
            self._current_batch_fn = fn
            self._current_batch_start = batch_offset
            self._current_batches = create_heterogeneous_sampled_graphs_from_dataframe(
                sampled_df=df,
                num_nodes_dict=self.num_nodes_dict,
                etype_id_dict=self.etype_id_dict,
                etype_offset_dict=self.etype_offset_dict,
                ntype_offset_dict=self.ntype_offset_dict,
                edge_dir=self.edge_dir,
            )
            del df

        current_offset = idx - batch_offset
        return self._current_batches[current_offset]

    def set_input_directory(self, input_directory):
        self._input_directory = input_directory
        self._sampled_files = os.listdir(input_directory)
        self._batch_to_fn_d = {
            i: get_batch_fn_batch_start(i, self._sampled_files)
            for i in range(0, self.num_batches + 1)
        }


def get_batch_fn_batch_start(batch_id, output_files):
    for fn in output_files:
        batch_start = fn.split("batch=")[1].split("-")[0]
        batch_start = int(batch_start)
        batch_end = fn.split("-")[1].split(".")[0]
        batch_end = int(batch_end)
        if batch_start <= batch_id and batch_id <= batch_end:
            return fn, batch_start

    raise ValueError(f"batch_id {id} not found in output_files: {output_files}")
