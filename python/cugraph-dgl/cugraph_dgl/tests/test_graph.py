# Copyright (c) 2024, NVIDIA CORPORATION.
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

import cugraph_dgl
import pylibcugraph
import cupy
import numpy as np

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
dgl = import_optional("dgl")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_homogeneous_graph(direction):
    df = karate.get_edgelist()
    df.src = df.src.astype("int64")
    df.dst = df.dst.astype("int64")
    wgt = np.random.random((len(df),))

    graph = cugraph_dgl.Graph()
    num_nodes = max(df.src.max(), df.dst.max()) + 1
    node_x = np.random.random((num_nodes,))

    graph.add_nodes(
        num_nodes, data={"num": torch.arange(num_nodes, dtype=torch.int64), "x": node_x}
    )
    graph.add_edges(df.src, df.dst, {"weight": wgt})
    plc_dgl_graph = graph._graph(direction=direction)

    assert graph.num_nodes() == num_nodes
    assert graph.num_edges() == len(df)
    assert graph.is_homogeneous
    assert not graph.is_multi_gpu

    assert (
        graph.nodes() == torch.arange(num_nodes, dtype=torch.int64, device="cuda")
    ).all()
    assert (graph.nodes[None]["x"] == torch.as_tensor(node_x, device="cuda")).all()
    assert (
        graph.nodes[None]["num"]
        == torch.arange(num_nodes, dtype=torch.int64, device="cuda")
    ).all()

    assert (
        graph.edges("eid", device="cuda")
        == torch.arange(len(df), dtype=torch.int64, device="cuda")
    ).all()
    assert (graph.edges[None]["weight"] == torch.as_tensor(wgt, device="cuda")).all()

    plc_expected_graph = pylibcugraph.SGGraph(
        pylibcugraph.ResourceHandle(),
        pylibcugraph.GraphProperties(is_multigraph=True, is_symmetric=False),
        df.src if direction == "out" else df.dst,
        df.dst if direction == "out" else df.src,
        vertices_array=cupy.arange(num_nodes, dtype="int64"),
    )

    # Do the expensive check to make sure this test fails if an invalid
    # graph is constructed.
    v_actual, d_in_actual, d_out_actual = pylibcugraph.degrees(
        pylibcugraph.ResourceHandle(),
        plc_dgl_graph,
        source_vertices=cupy.arange(num_nodes, dtype="int64"),
        do_expensive_check=True,
    )

    v_exp, d_in_exp, d_out_exp = pylibcugraph.degrees(
        pylibcugraph.ResourceHandle(),
        plc_expected_graph,
        source_vertices=cupy.arange(num_nodes, dtype="int64"),
        do_expensive_check=True,
    )

    assert (v_actual == v_exp).all()
    assert (d_in_actual == d_in_exp).all()
    assert (d_out_actual == d_out_exp).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_heterogeneous_graph(direction):
    df = karate.get_edgelist()
    df.src = df.src.astype("int64")
    df.dst = df.dst.astype("int64")

    graph = cugraph_dgl.Graph()
    total_num_nodes = max(df.src.max(), df.dst.max()) + 1

    num_nodes_group_1 = total_num_nodes // 2
    num_nodes_group_2 = total_num_nodes - num_nodes_group_1

    node_x_1 = np.random.random((num_nodes_group_1,))
    node_x_2 = np.random.random((num_nodes_group_2,))

    graph.add_nodes(num_nodes_group_1, {"x": node_x_1}, "type1")
    graph.add_nodes(num_nodes_group_2, {"x": node_x_2}, "type2")

    edges_11 = df[(df.src < num_nodes_group_1) & (df.dst < num_nodes_group_1)]
    edges_12 = df[(df.src < num_nodes_group_1) & (df.dst >= num_nodes_group_1)]
    edges_21 = df[(df.src >= num_nodes_group_1) & (df.dst < num_nodes_group_1)]
    edges_22 = df[(df.src >= num_nodes_group_1) & (df.dst >= num_nodes_group_1)]

    edges_12.dst -= num_nodes_group_1
    edges_21.src -= num_nodes_group_1
    edges_22.dst -= num_nodes_group_1
    edges_22.src -= num_nodes_group_1

    graph.add_edges(edges_11.src, edges_11.dst, etype=("type1", "e1", "type1"))
    graph.add_edges(edges_12.src, edges_12.dst, etype=("type1", "e2", "type2"))
    graph.add_edges(edges_21.src, edges_21.dst, etype=("type2", "e3", "type1"))
    graph.add_edges(edges_22.src, edges_22.dst, etype=("type2", "e4", "type2"))

    assert not graph.is_homogeneous
    assert not graph.is_multi_gpu

    # Verify graph.nodes()
    assert (
        graph.nodes() == torch.arange(total_num_nodes, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.nodes("type1")
        == torch.arange(num_nodes_group_1, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.nodes("type2")
        == torch.arange(num_nodes_group_2, dtype=torch.int64, device="cuda")
    ).all()

    # Verify graph.edges()
    assert (
        graph.edges("eid", etype=("type1", "e1", "type1"))
        == torch.arange(len(edges_11), dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.edges("eid", etype=("type1", "e2", "type2"))
        == torch.arange(len(edges_12), dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.edges("eid", etype=("type2", "e3", "type1"))
        == torch.arange(len(edges_21), dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.edges("eid", etype=("type2", "e4", "type2"))
        == torch.arange(len(edges_22), dtype=torch.int64, device="cuda")
    ).all()

    # Use sampling call to check graph creation
    # This isn't a test of cuGraph sampling with DGL; the options are
    # set to verify the graph only.
    plc_graph = graph._graph(direction)
    sampling_output = pylibcugraph.uniform_neighbor_sample(
        pylibcugraph.ResourceHandle(),
        plc_graph,
        start_list=cupy.arange(total_num_nodes, dtype="int64"),
        h_fan_out=np.array([1, 1], dtype="int32"),
        with_replacement=False,
        do_expensive_check=True,
        with_edge_properties=True,
        prior_sources_behavior="exclude",
        return_dict=True,
    )

    expected_etypes = {
        0: "e1",
        1: "e2",
        2: "e3",
        3: "e4",
    }
    expected_offsets = {
        0: (0, 0),
        1: (0, num_nodes_group_1),
        2: (num_nodes_group_1, 0),
        3: (num_nodes_group_1, num_nodes_group_1),
    }
    if direction == "in":
        src_col = "minors"
        dst_col = "majors"
    else:
        src_col = "majors"
        dst_col = "minors"

    # Looping over the output verifies that all edges are valid
    # (and therefore, the graph is valid)
    for i, etype in enumerate(sampling_output["edge_type"].tolist()):
        eid = int(sampling_output["edge_id"][i])

        srcs, dsts, eids = graph.edges(
            "all", etype=expected_etypes[etype], device="cpu"
        )

        assert eids[eid] == eid
        assert (
            srcs[eid] == int(sampling_output[src_col][i]) - expected_offsets[etype][0]
        )
        assert (
            dsts[eid] == int(sampling_output[dst_col][i]) - expected_offsets[etype][1]
        )
