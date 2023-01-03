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

from cugraph.gnn.dgl_extensions.utils.add_data import (
    add_edge_data_from_parquet,
    add_node_data_from_parquet,
)


def add_node_data_from_parquet_remote(
    file_path,
    node_col_name,
    node_offset,
    ntype,
    feat_name,
    contains_vector_features,
    graph_id,
    server,
):
    pG = server.get_graph(graph_id)

    add_node_data_from_parquet(
        file_path=file_path,
        node_col_name=node_col_name,
        node_offset=node_offset,
        ntype=ntype,
        feat_name=feat_name,
        contains_vector_features=contains_vector_features,
        pG=pG,
    )
    return


def add_edge_data_from_parquet_remote(
    file_path,
    node_col_names,
    canonical_etype,
    src_offset,
    dst_offset,
    feat_name,
    contains_vector_features,
    graph_id,
    server,
):
    pG = server.get_graph(graph_id)

    add_edge_data_from_parquet(
        file_path=file_path,
        node_col_names=node_col_names,
        canonical_etype=canonical_etype,
        src_offset=src_offset,
        dst_offset=dst_offset,
        feat_name=feat_name,
        contains_vector_features=contains_vector_features,
        pG=pG,
    )
    return
