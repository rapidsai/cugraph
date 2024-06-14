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


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
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
