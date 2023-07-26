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

import cudf
import cupy

import pytest

from cugraph_pyg.data import CuGraphStore
from cugraph_pyg.sampler.cugraph_sampler import _sampler_output_from_sampling_results

from cugraph.utilities.utils import import_optional, MissingModule
from cugraph import uniform_neighbor_sample

torch = import_optional("torch")


@pytest.mark.cugraph_ops
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_neighbor_sample(basic_graph_1):
    F, G, N = basic_graph_1
    cugraph_store = CuGraphStore(F, G, N)

    sampling_results = uniform_neighbor_sample(
        cugraph_store._subgraph(),
        cudf.Series([0, 1, 2, 3, 4], dtype="int64"),
        fanout_vals=[-1],
        with_replacement=False,
        with_edge_properties=True,
        batch_id_list=cudf.Series(cupy.zeros(5, dtype="int32")),
        random_state=62,
        return_offsets=False,
    ).sort_values(by=["sources", "destinations"])

    out = _sampler_output_from_sampling_results(
        sampling_results=sampling_results,
        renumber_map=None,
        graph_store=cugraph_store,
        metadata=torch.arange(6, dtype=torch.int64),
    )

    noi_groups = out.node
    row_dict = out.row
    col_dict = out.col
    metadata = out.metadata

    assert metadata.tolist() == list(range(6))

    for node_type, node_ids in noi_groups.items():
        actual_vertex_ids = torch.arange(N[node_type])

        assert sorted(node_ids.tolist()) == actual_vertex_ids.tolist()

    assert (
        row_dict[("vt1", "pig", "vt1")].tolist() == G[("vt1", "pig", "vt1")][0].tolist()
    )
    assert (
        col_dict[("vt1", "pig", "vt1")].tolist() == G[("vt1", "pig", "vt1")][1].tolist()
    )

    # check the hop dictionaries
    assert len(out.num_sampled_nodes) == 1
    assert out.num_sampled_nodes["vt1"].tolist() == [4, 4]

    assert len(out.num_sampled_edges) == 1
    assert out.num_sampled_edges[("vt1", "pig", "vt1")].tolist() == [6]


@pytest.mark.cugraph_ops
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_neighbor_sample_multi_vertex(multi_edge_multi_vertex_graph_1):
    F, G, N = multi_edge_multi_vertex_graph_1
    cugraph_store = CuGraphStore(F, G, N)

    sampling_results = uniform_neighbor_sample(
        cugraph_store._subgraph(),
        cudf.Series([0, 1, 2, 3, 4], dtype="int64"),
        fanout_vals=[-1],
        with_replacement=False,
        with_edge_properties=True,
        batch_id_list=cudf.Series(cupy.zeros(5, dtype="int32")),
        random_state=62,
        return_offsets=False,
    ).sort_values(by=["sources", "destinations"])

    out = _sampler_output_from_sampling_results(
        sampling_results=sampling_results,
        renumber_map=None,
        graph_store=cugraph_store,
        metadata=torch.arange(6, dtype=torch.int64),
    )

    noi_groups = out.node
    row_dict = out.row
    col_dict = out.col
    metadata = out.metadata

    assert metadata.tolist() == list(range(6))

    for node_type, node_ids in noi_groups.items():
        actual_vertex_ids = torch.arange(N[node_type])

        assert node_ids.tolist() == sorted(actual_vertex_ids.tolist())

    for edge_type, ei in G.items():
        assert sorted(row_dict[edge_type].tolist()) == sorted(ei[0].tolist())
        assert sorted(col_dict[edge_type].tolist()) == sorted(ei[1].tolist())

    # check the hop dictionaries
    assert len(out.num_sampled_nodes) == 2
    assert out.num_sampled_nodes["black"].tolist() == [2, 2]
    assert out.num_sampled_nodes["brown"].tolist() == [3, 2]

    assert len(out.num_sampled_edges) == 5
    assert out.num_sampled_edges[("brown", "horse", "brown")].tolist() == [2]
    assert out.num_sampled_edges[("brown", "tortoise", "black")].tolist() == [3]
    assert out.num_sampled_edges[("brown", "mongoose", "black")].tolist() == [2]
    assert out.num_sampled_edges[("black", "cow", "brown")].tolist() == [2]
    assert out.num_sampled_edges[("black", "snake", "black")].tolist() == [1]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_neighbor_sample_mock_sampling_results(abc_graph):
    F, G, N = abc_graph

    graph_store = CuGraphStore(F, G, N)

    # let 0, 1 be the start vertices, fanout = [2, 1, 2, 3]
    mock_sampling_results = cudf.DataFrame(
        {
            "sources": cudf.Series([0, 0, 1, 2, 3, 3, 1, 3, 3, 3], dtype="int64"),
            "destinations": cudf.Series([2, 3, 3, 8, 1, 7, 3, 1, 5, 7], dtype="int64"),
            "hop_id": cudf.Series([0, 0, 0, 1, 1, 1, 2, 3, 3, 3], dtype="int32"),
            "edge_type": cudf.Series([0, 0, 0, 2, 1, 2, 0, 1, 2, 2], dtype="int32"),
        }
    )

    out = _sampler_output_from_sampling_results(
        mock_sampling_results, None, graph_store, None
    )

    assert out.metadata is None
    assert len(out.node) == 3
    assert out.node["A"].tolist() == [0, 1]
    assert out.node["B"].tolist() == [0, 1]
    assert out.node["C"].tolist() == [3, 2, 0]

    assert len(out.row) == 3
    assert len(out.col) == 3
    assert out.row[("A", "ab", "B")].tolist() == [0, 0, 1, 1]
    assert out.col[("A", "ab", "B")].tolist() == [0, 1, 1, 1]
    assert out.row[("B", "bc", "C")].tolist() == [0, 1, 1, 1]
    assert out.col[("B", "bc", "C")].tolist() == [0, 1, 2, 1]
    assert out.row[("B", "ba", "A")].tolist() == [1, 1]
    assert out.col[("B", "ba", "A")].tolist() == [1, 1]

    assert len(out.num_sampled_nodes) == 3
    assert out.num_sampled_nodes["A"].tolist() == [2, 0, 1, 0, 1]
    assert out.num_sampled_nodes["B"].tolist() == [0, 2, 0, 1, 0]
    assert out.num_sampled_nodes["C"].tolist() == [0, 0, 2, 0, 2]

    assert len(out.num_sampled_edges) == 3
    assert out.num_sampled_edges[("A", "ab", "B")].tolist() == [3, 0, 1, 0]
    assert out.num_sampled_edges[("B", "ba", "A")].tolist() == [0, 1, 0, 1]
    assert out.num_sampled_edges[("B", "bc", "C")].tolist() == [0, 2, 0, 2]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skip("needs to be written")
def test_neighbor_sample_renumbered():
    pass
