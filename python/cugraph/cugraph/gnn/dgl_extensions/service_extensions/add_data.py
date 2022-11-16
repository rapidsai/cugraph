# Copyright (c) 2022, NVIDIA CORPORATION.
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

import cupy as cp
from cugraph.gnn.dgl_extensions.utils.add_data import (
    add_edge_data_from_parquet,
    add_node_data_from_parquet,
)


def add_node_data_from_parquet_remote(
    file_path, node_col_name, node_offset, ntype, graph_id, server
):
    pG = server.get_graph(graph_id)

    columns_list = add_node_data_from_parquet(
        file_path, node_col_name, node_offset, ntype, pG
    )
    return serialize_strings_to_array(columns_list)


def add_edge_data_from_parquet_remote(
    file_path, node_col_names, canonical_etype, src_offset, dst_offset, graph_id, server
):
    pG = server.get_graph(graph_id)

    columns_list = add_edge_data_from_parquet(
        file_path, node_col_names, canonical_etype, src_offset, dst_offset, pG
    )
    return serialize_strings_to_array(columns_list)


def convert_to_string_ar(string):
    return cp.asarray([ord(c) for c in string], cp.int32), len(string)


def serialize_strings_to_array(strings_list):
    ar_ls = []
    len_ls = []
    for s in strings_list:
        ar, s_len = convert_to_string_ar(s)
        ar_ls.append(ar)
        len_ls.append(s_len)
    return cp.concatenate(ar_ls), cp.asarray(len_ls, dtype=cp.int32)
