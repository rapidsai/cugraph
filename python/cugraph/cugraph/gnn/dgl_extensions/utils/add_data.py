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

# Utils for adding data to cugraph graphstore objects
import dask_cudf
import cudf
from cugraph.experimental import MGPropertyGraph


def add_node_data_from_parquet(
    file_path,
    node_col_name,
    node_offset,
    ntype,
    feat_name,
    contains_vector_features,
    pG,
):
    if isinstance(pG, MGPropertyGraph):
        df = dask_cudf.read_parquet(file_path)
    else:
        df = cudf.read_parquet(file_path)

    df[node_col_name] = df[node_col_name] + node_offset
    add_node_data_from_dataframe(
        df, node_col_name, ntype, feat_name, contains_vector_features, pG
    )
    return


def add_node_data_from_dataframe(
    df, node_col_name, ntype, feat_name, contains_vector_features, pG
):
    if contains_vector_features:
        if feat_name is None:
            raise ValueError(
                "feature name must be provided when wrapping"
                + " multiple columns under a single feature name"
                + " or a feature map"
            )
        elif isinstance(feat_name, dict):
            pG.add_vertex_data(
                df,
                vertex_col_name=node_col_name,
                type_name=ntype,
                vector_properties=feat_name,
            )
        else:
            pG.add_vertex_data(
                df,
                vertex_col_name=node_col_name,
                type_name=ntype,
                vector_property=feat_name,
            )
    else:
        if feat_name is not None:
            raise ValueError(
                f"feat_name {feat_name} is only valid when "
                "wrapping multiple columns under feature names"
            )
        pG.add_vertex_data(df, vertex_col_name=node_col_name, type_name=ntype)


def add_edge_data_from_parquet(
    file_path,
    node_col_names,
    canonical_etype,
    src_offset,
    dst_offset,
    feat_name,
    contains_vector_features,
    pG,
):
    if isinstance(pG, MGPropertyGraph):
        df = dask_cudf.read_parquet(file_path)
    else:
        df = cudf.read_parquet(file_path)

    df[node_col_names[0]] = df[node_col_names[0]] + src_offset
    df[node_col_names[1]] = df[node_col_names[1]] + dst_offset
    add_edge_data_from_dataframe(
        df, node_col_names, canonical_etype, feat_name, contains_vector_features, pG
    )

    return


def add_edge_data_from_dataframe(
    df, node_col_names, canonical_etype, feat_name, contains_vector_features, pG
):
    if contains_vector_features:
        if feat_name is None:
            raise ValueError(
                "feature name must be provided when wrapping"
                + " multiple columns under a single feature name"
                + " or a feature map"
            )
        elif isinstance(feat_name, dict):
            pG.add_edge_data(
                df,
                vertex_col_names=node_col_names,
                type_name=canonical_etype,
                vector_properties=feat_name,
            )
        else:
            pG.add_edge_data(
                df,
                vertex_col_names=node_col_names,
                type_name=canonical_etype,
                vector_property=feat_name,
            )
    else:
        if feat_name is not None:
            raise ValueError(
                f"feat_name {feat_name} is only valid when "
                "wrapping multiple columns under feature names"
            )
        pG.add_edge_data(df, vertex_col_names=node_col_names, type_name=canonical_etype)
