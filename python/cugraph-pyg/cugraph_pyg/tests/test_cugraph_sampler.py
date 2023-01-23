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

from cugraph_pyg.data import to_pyg
from cugraph_pyg.sampler import CuGraphSampler
from cugraph.experimental import PropertyGraph

import cudf
import cupy

import pytest


@pytest.fixture
def basic_property_graph_1():
    pG = PropertyGraph()
    pG.add_edge_data(
        cudf.DataFrame({"src": [0, 0, 1, 2, 2, 3], "dst": [1, 2, 4, 3, 4, 1]}),
        vertex_col_names=["src", "dst"],
        type_name="pig",
    )

    pG.add_vertex_data(
        cudf.DataFrame(
            {
                "prop1": [100, 200, 300, 400, 500],
                "prop2": [5, 4, 3, 2, 1],
                "id": [0, 1, 2, 3, 4],
            }
        ),
        vertex_col_name="id",
    )

    return pG


@pytest.fixture
def multi_edge_multi_vertex_property_graph_1():
    df = cudf.DataFrame(
        {
            "src": [0, 0, 1, 2, 2, 3, 3, 1, 2, 4],
            "dst": [1, 2, 4, 3, 3, 1, 2, 4, 4, 3],
            "edge_type": [
                "horse",
                "horse",
                "duck",
                "duck",
                "mongoose",
                "cow",
                "cow",
                "mongoose",
                "duck",
                "snake",
            ],
        }
    )

    pG = PropertyGraph()
    for edge_type in df.edge_type.unique().to_pandas():
        pG.add_edge_data(
            df[df.edge_type == edge_type],
            vertex_col_names=["src", "dst"],
            type_name=edge_type,
        )

    vdf = cudf.DataFrame(
        {
            "prop1": [100, 200, 300, 400, 500],
            "prop2": [5, 4, 3, 2, 1],
            "id": [0, 1, 2, 3, 4],
            "vertex_type": [
                "brown",
                "brown",
                "brown",
                "black",
                "black",
            ],
        }
    )

    for vertex_type in vdf.vertex_type.unique().to_pandas():
        vd = vdf[vdf.vertex_type == vertex_type].drop("vertex_type", axis=1)
        pG.add_vertex_data(vd, vertex_col_name="id", type_name=vertex_type)

    return pG


@pytest.mark.cugraph_ops
@pytest.mark.skip(reason="deprecated API")
def test_neighbor_sample(basic_property_graph_1):
    pG = basic_property_graph_1
    feature_store, graph_store = to_pyg(pG, backend="cupy")
    sampler = CuGraphSampler(
        (feature_store, graph_store),
        num_neighbors=[-1],
        replace=True,
        directed=True,
        edge_types=[v.edge_type for v in graph_store._edge_types_to_attrs.values()],
    )

    out_dict = sampler.sample_from_nodes(
        (
            cupy.arange(6, dtype="int32"),
            cupy.array([0, 1, 2, 3, 4], dtype="int32"),
            None,
        )
    )

    if isinstance(out_dict, dict):
        noi_groups, row_dict, col_dict, _ = out_dict["out"]
        metadata = out_dict["metadata"]
    else:
        noi_groups = out_dict.node
        row_dict = out_dict.row
        col_dict = out_dict.col
        metadata = out_dict.metadata

    assert metadata.get().tolist() == list(range(6))

    for node_type, node_ids in noi_groups.items():
        actual_vertex_ids = pG.get_vertex_data(types=[node_type])[
            pG.vertex_col_name
        ].to_cupy()

        assert list(node_ids) == list(actual_vertex_ids)

    cols = [pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    combined_df = cudf.DataFrame()
    for edge_type, row in row_dict.items():
        col = col_dict[edge_type]
        df = cudf.DataFrame({pG.src_col_name: row, pG.dst_col_name: col})
        df[pG.type_col_name] = edge_type[1]
        combined_df = cudf.concat([combined_df, df])
    combined_df = combined_df.sort_values(cols)
    combined_df = combined_df.reset_index().drop("index", axis=1)

    base_df = pG.get_edge_data()
    base_df = base_df[cols]
    base_df = base_df.sort_values(cols)
    base_df = base_df.reset_index().drop("index", axis=1)

    assert (
        combined_df.drop_duplicates().values_host.tolist()
        == base_df.values_host.tolist()
    )


@pytest.mark.cugraph_ops
@pytest.mark.skip(reason="deprecated API")
def test_neighbor_sample_multi_vertex(multi_edge_multi_vertex_property_graph_1):
    pG = multi_edge_multi_vertex_property_graph_1
    feature_store, graph_store = to_pyg(pG, backend="cupy")
    sampler = CuGraphSampler(
        (feature_store, graph_store),
        num_neighbors=[-1],
        replace=True,
        directed=True,
        edge_types=[v.edge_type for v in graph_store._edge_types_to_attrs.values()],
    )

    out_dict = sampler.sample_from_nodes(
        (
            cupy.arange(6, dtype="int32"),
            cupy.array([0, 1, 2, 3, 4], dtype="int32"),
            None,
        )
    )

    if isinstance(out_dict, dict):
        _, row_dict, col_dict, _ = out_dict["out"]
        metadata = out_dict["metadata"]
    else:
        row_dict = out_dict.row
        col_dict = out_dict.col
        metadata = out_dict.metadata

    assert metadata.get().tolist() == list(range(6))

    for pyg_can_edge_type, srcs in row_dict.items():
        dsts = col_dict[pyg_can_edge_type]
        num_unique_sampled_edges = len(
            cudf.DataFrame({"src": srcs, "dst": dsts}).drop_duplicates()
        )

        cugraph_edge_type = pyg_can_edge_type[1]
        num_edges = len(pG.get_edge_data(types=[cugraph_edge_type]))
        assert num_edges == num_unique_sampled_edges
