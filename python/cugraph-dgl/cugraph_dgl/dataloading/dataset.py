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
from typing import Tuple, Dict, Optional, List, Union

import os
import cudf
from cugraph.utilities.utils import import_optional
from cugraph_dgl.dataloading.utils.sampling_helpers import (
    create_homogeneous_sampled_graphs_from_dataframe,
    create_heterogeneous_sampled_graphs_from_dataframe,
    create_homogeneous_sampled_graphs_from_dataframe_csc,
)


dgl = import_optional("dgl")
torch = import_optional("torch")


# Todo: maybe should switch to __iter__
class HomogenousBulkSamplerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        total_number_of_nodes: int,
        edge_dir: str,
        return_type: str = "dgl.Block",
        sparse_format: str = "coo",
    ):
        if return_type not in ["dgl.Block", "cugraph_dgl.nn.SparseGraph"]:
            raise ValueError(
                "return_type must be either 'dgl.Block' or "
                "'cugraph_dgl.nn.SparseGraph'."
            )
        # TODO: Deprecate `total_number_of_nodes`
        # as it is no longer needed
        # in the next release
        self.total_number_of_nodes = total_number_of_nodes
        self.edge_dir = edge_dir
        self.sparse_format = sparse_format
        self._current_batch_fn = None
        self._input_files = None
        self._return_type = return_type

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx: int):
        if self._input_files is None:
            raise dgl.DGLError(
                "Please set input files by calling `set_input_files` "
                "before trying to fetch a sample"
            )

        fn, batch_offset = self._batch_to_fn_d[idx]
        if fn != self._current_batch_fn:
            if self.sparse_format == "csc":
                df = _load_sampled_file(dataset_obj=self, fn=fn, skip_rename=True)
                self._current_batches = (
                    create_homogeneous_sampled_graphs_from_dataframe_csc(df)
                )
            else:
                df = _load_sampled_file(dataset_obj=self, fn=fn)
                self._current_batches = (
                    create_homogeneous_sampled_graphs_from_dataframe(
                        sampled_df=df,
                        edge_dir=self.edge_dir,
                        return_type=self._return_type,
                    )
                )
        current_offset = idx - batch_offset
        return self._current_batches[current_offset]

    def set_input_files(
        self,
        input_directory: Optional[str] = None,
        input_file_paths: Optional[List[str]] = None,
    ):
        """
        Set input files that have been created by the `cugraph.gnn.BulkSampler`
        Parameters
        ----------
        input_directory: str
           input_directory which contains all the files that will be
           loaded by HomogenousBulkSamplerDataset
        input_file_paths: List[str]
            File paths that will be loaded by the HomogenousBulkSamplerDataset
        """
        _set_input_files(
            self, input_directory=input_directory, input_file_paths=input_file_paths
        )


class HeterogenousBulkSamplerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        etype_id_dict: Dict[int, Tuple[str, str, str]],
        etype_offset_dict: Dict[Tuple[str, str, str], int],
        ntype_offset_dict: Dict[str, int],
        edge_dir: str = "in",
    ):
        self.num_nodes_dict = num_nodes_dict
        self.etype_id_dict = etype_id_dict
        self.etype_offset_dict = etype_offset_dict
        self.ntype_offset_dict = ntype_offset_dict
        self.edge_dir = edge_dir
        self._current_batch_fn = None
        self._input_files = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self._input_files is None:
            raise dgl.DGLError(
                "Please set input files by calling `set_input_files` "
                "before trying to fetch a sample"
            )

        fn, batch_offset = self._batch_to_fn_d[idx]
        if fn != self._current_batch_fn:
            df = _load_sampled_file(dataset_obj=self, fn=fn)
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

    def set_input_files(
        self,
        input_directory: Optional[str] = None,
        input_file_paths: Optional[List[str]] = None,
    ):
        """
        Set input files that have been created by the `cugraph.gnn.BulkSampler`
        Parameters
        ----------
        input_directory: str
            input_directory which contains all the files that will be
            loaded by HeterogenousBulkSamplerDataset
        input_file_paths: List[str]
            File names that will be loaded by the HeterogenousBulkSamplerDataset
        """
        _set_input_files(
            self, input_directory=input_directory, input_file_paths=input_file_paths
        )


def _load_sampled_file(dataset_obj, fn, skip_rename=False):
    df = cudf.read_parquet(os.path.join(fn))
    if dataset_obj.edge_dir == "in" and not skip_rename:
        df.rename(
            columns={"sources": "destinations", "destinations": "sources"},
            inplace=True,
        )
    dataset_obj._current_batch_fn = fn
    return df


def get_batch_start_end(fn):
    batch_str = fn.split("batch=")[1]
    batch_start, batch_end = batch_str.split("-")
    batch_end = batch_end.split(".parquet")[0]
    return int(batch_start), int(batch_end)


def get_batch_to_fn_d(files):
    batch_to_fn_d = {}
    batch_id = 0
    for fn in files:
        start, end = get_batch_start_end(fn)
        batch_offset = batch_id
        for _ in range(start, end + 1):
            batch_to_fn_d[batch_id] = fn, batch_offset
            batch_id += 1
    return batch_to_fn_d


def _set_input_files(
    dataset_obj: Union[HomogenousBulkSamplerDataset, HeterogenousBulkSamplerDataset],
    input_directory: Optional[str] = None,
    input_file_paths: Optional[List[str]] = None,
) -> None:

    if input_directory is None and input_file_paths is None:
        raise ValueError("input_files or input_file_paths must be set")

    if (input_directory is not None) and (input_file_paths is not None):
        raise ValueError("Only one of input_directory or input_file_paths must be set")

    if input_file_paths is not None:
        dataset_obj._input_files = input_file_paths
    if input_directory is not None:
        dataset_obj._input_files = [fp.path for fp in os.scandir(input_directory)]
    dataset_obj._batch_to_fn_d = get_batch_to_fn_d(dataset_obj._input_files)
    dataset_obj.num_batches = len(dataset_obj._batch_to_fn_d)
    dataset_obj._current_batch_fn = None
