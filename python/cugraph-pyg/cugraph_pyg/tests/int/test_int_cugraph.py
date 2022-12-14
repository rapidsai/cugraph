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

import pytest

# Integration test requirements
"""
rmm
cugraph
cugraph_pyg
cudf
dask_cudf
ogb
torch_geometric
"""
import cudf
import dask_cudf
from cugraph.experimental import PropertyGraph, MGPropertyGraph
from ogb.nodeproppred import NodePropPredDataset

from cugraph_pyg.data import to_pyg
from cugraph_pyg.sampler import CuGraphSampler

from torch_geometric.loader import NodeLoader


@pytest.fixture(scope="module")
def loader_hetero_mag():
    # Load MAG into CPU memory
    dataset = NodePropPredDataset(name="ogbn-mag")

    data = dataset[0]
    pG = PropertyGraph()

    # Load the vertex ids into a new property graph
    vertex_offsets = {}
    last_offset = 0

    for node_type, num_nodes in data[0]["num_nodes_dict"].items():
        vertex_offsets[node_type] = last_offset
        last_offset += num_nodes

        blank_df = cudf.DataFrame(
            {
                "id": range(
                    vertex_offsets[node_type], vertex_offsets[node_type] + num_nodes
                )
            }
        )
        blank_df.id = blank_df.id.astype("int64")

        pG.add_vertex_data(blank_df, vertex_col_name="id", type_name=node_type)

    # Add the remaining vertex features
    for i, (node_type, node_features) in enumerate(data[0]["node_feat_dict"].items()):
        vertex_offset = vertex_offsets[node_type]

        feature_df = cudf.DataFrame(node_features)
        feature_df.columns = [str(c) for c in range(feature_df.shape[1])]
        feature_df["id"] = range(vertex_offset, vertex_offset + node_features.shape[0])
        feature_df.id = feature_df.id.astype("int64")

        pG.add_vertex_data(feature_df, vertex_col_name="id", type_name=node_type)

    # Fill in an empty value for vertices without properties.
    pG.fillna(0.0)

    # Add the edges
    for i, (edge_key, eidx) in enumerate(data[0]["edge_index_dict"].items()):
        node_type_src, edge_type, node_type_dst = edge_key
        print(node_type_src, edge_type, node_type_dst)
        vertex_offset_src = vertex_offsets[node_type_src]
        vertex_offset_dst = vertex_offsets[node_type_dst]
        eidx = [n + vertex_offset_src for n in eidx[0]], [
            n + vertex_offset_dst for n in eidx[1]
        ]

        edge_df = cudf.DataFrame({"src": eidx[0], "dst": eidx[1]})
        edge_df.src = edge_df.src.astype("int64")
        edge_df.dst = edge_df.dst.astype("int64")
        edge_df["type"] = edge_type

        # Adding backwards edges is currently required in both
        # the cuGraph PG and PyG APIs.
        pG.add_edge_data(edge_df, vertex_col_names=["src", "dst"], type_name=edge_type)
        pG.add_edge_data(
            edge_df, vertex_col_names=["dst", "src"], type_name=f"{edge_type}_bw"
        )

    # Add the target variable
    y_df = cudf.DataFrame(data[1]["paper"], columns=["y"])
    y_df["id"] = range(vertex_offsets["paper"], vertex_offsets["paper"] + len(y_df))
    y_df.id = y_df.id.astype("int64")

    pG.add_vertex_data(y_df, vertex_col_name="id", type_name="paper")

    # Construct a graph/feature store and loaders
    feature_store, graph_store = to_pyg(pG)
    sampler = CuGraphSampler(
        data=(feature_store, graph_store),
        shuffle=True,
        num_neighbors=[10, 25],
        batch_size=50,
    )
    loader = NodeLoader(
        data=(feature_store, graph_store),
        shuffle=True,
        batch_size=50,
        node_sampler=sampler,
        input_nodes=("author", graph_store.get_vertex_index("author")),
    )

    return loader


@pytest.fixture(scope="module")
def loader_hetero_mag_multi_gpu(rmmc):
    # Load MAG into CPU memory
    dataset = NodePropPredDataset(name="ogbn-mag")

    data = dataset[0]
    pG = MGPropertyGraph()

    # Load the vertex ids into a new property graph
    vertex_offsets = {}
    last_offset = 0

    for node_type, num_nodes in data[0]["num_nodes_dict"].items():
        vertex_offsets[node_type] = last_offset
        last_offset += num_nodes

        blank_df = cudf.DataFrame(
            {
                "id": range(
                    vertex_offsets[node_type], vertex_offsets[node_type] + num_nodes
                )
            }
        )
        blank_df.id = blank_df.id.astype("int64")
        blank_df = dask_cudf.from_cudf(blank_df, npartitions=2)

        pG.add_vertex_data(blank_df, vertex_col_name="id", type_name=node_type)

    # Add the remaining vertex features
    for i, (node_type, node_features) in enumerate(data[0]["node_feat_dict"].items()):
        vertex_offset = vertex_offsets[node_type]

        feature_df = cudf.DataFrame(node_features)
        feature_df.columns = [str(c) for c in range(feature_df.shape[1])]
        feature_df["id"] = range(vertex_offset, vertex_offset + node_features.shape[0])
        feature_df.id = feature_df.id.astype("int64")
        feature_df = dask_cudf.from_cudf(feature_df, npartitions=2)

        pG.add_vertex_data(feature_df, vertex_col_name="id", type_name=node_type)

    # Fill in an empty value for vertices without properties.
    pG.fillna(0.0)

    # Add the edges
    for i, (edge_key, eidx) in enumerate(data[0]["edge_index_dict"].items()):
        node_type_src, edge_type, node_type_dst = edge_key
        print(node_type_src, edge_type, node_type_dst)
        vertex_offset_src = vertex_offsets[node_type_src]
        vertex_offset_dst = vertex_offsets[node_type_dst]
        eidx = [n + vertex_offset_src for n in eidx[0]], [
            n + vertex_offset_dst for n in eidx[1]
        ]

        edge_df = cudf.DataFrame({"src": eidx[0], "dst": eidx[1]})
        edge_df.src = edge_df.src.astype("int64")
        edge_df.dst = edge_df.dst.astype("int64")
        edge_df["type"] = edge_type
        edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)

        # Adding backwards edges is currently required in both
        # the cuGraph PG and PyG APIs.
        pG.add_edge_data(edge_df, vertex_col_names=["src", "dst"], type_name=edge_type)
        pG.add_edge_data(
            edge_df, vertex_col_names=["dst", "src"], type_name=f"{edge_type}_bw"
        )

    # Add the target variable
    y_df = cudf.DataFrame(data[1]["paper"], columns=["y"])
    y_df["id"] = range(vertex_offsets["paper"], vertex_offsets["paper"] + len(y_df))
    y_df.id = y_df.id.astype("int64")
    y_df = dask_cudf.from_cudf(y_df, npartitions=2)

    pG.add_vertex_data(y_df, vertex_col_name="id", type_name="paper")

    # Construct a graph/feature store and loaders
    feature_store, graph_store = to_pyg(pG)
    sampler = CuGraphSampler(
        data=(feature_store, graph_store),
        shuffle=True,
        num_neighbors=[10, 25],
        batch_size=50,
    )
    loader = NodeLoader(
        data=(feature_store, graph_store),
        shuffle=True,
        batch_size=50,
        node_sampler=sampler,
        input_nodes=("author", graph_store.get_vertex_index("author")),
    )

    return loader
