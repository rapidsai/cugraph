# Copyright (c) 2023, NVIDIA CORPORATION.
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
import networkx as nx
import pytest

import cugraph_nx as cnx


@pytest.mark.parametrize("graph_class", [nx.Graph, nx.DiGraph])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"preserve_edge_attrs": True},
        {"preserve_node_attrs": True},
        {"preserve_all_attrs": True},
        {"edge_attrs": {"x": 0}},
        {"edge_attrs": {"x": None}},
        {"edge_attrs": "x"},
        {"node_attrs": {"x": 0}},
        {"node_attrs": {"x": None}},
        {"node_attrs": "x"},
        {"weight": "x"},
    ],
)
def test_convert_empty(graph_class, kwargs):
    G = graph_class()
    cG = cnx.from_networkx(G, **kwargs)
    H = cnx.to_networkx(cG)
    assert G.number_of_nodes() == cG.number_of_nodes() == H.number_of_nodes() == 0
    assert G.number_of_edges() == cG.number_of_edges() == H.number_of_edges() == 0
    assert cG.edge_values == cG.edge_masks == cG.node_values == cG.node_masks == {}
    assert G.graph == cG.graph == H.graph == {}


def test_convert():
    G = nx.Graph()
    G.add_edge(0, 1, x=2)
    G.add_node(0, foo=10)
    G.add_node(1, foo=20, bar=100)
    for kwargs in [
        {"preserve_edge_attrs": True},
        {"preserve_all_attrs": True},
        {"edge_attrs": {"x": 0}},
        {"edge_attrs": {"x": None}},
        {"edge_attrs": "x", "edge_dtypes": int},
    ]:
        # All edges have "x" attribute, so all kwargs are equivalent
        cG = cnx.from_networkx(G, **kwargs)
        cp.testing.assert_array_equal(cG.row_indices, [0, 1])
        cp.testing.assert_array_equal(cG.col_indices, [1, 0])
        cp.testing.assert_array_equal(cG.edge_values["x"], [2, 2])
        assert len(cG.edge_values) == 1
        assert cG.edge_masks == {}
        H = cnx.to_networkx(cG)
        assert G.number_of_nodes() == cG.number_of_nodes() == H.number_of_nodes() == 2
        assert G.number_of_edges() == cG.number_of_edges() == H.number_of_edges() == 1
        assert G.adj == H.adj

    # Structure-only graph (no edge attributes)
    cG = cnx.from_networkx(G, preserve_node_attrs=True)
    cp.testing.assert_array_equal(cG.row_indices, [0, 1])
    cp.testing.assert_array_equal(cG.col_indices, [1, 0])
    cp.testing.assert_array_equal(cG.node_values["foo"], [10, 20])
    assert cG.edge_values == cG.edge_masks == {}
    H = cnx.to_networkx(cG)
    assert set(G.edges) == set(H.edges) == {(0, 1)}
    assert G.nodes == H.nodes

    # Fill completely missing attribute with default value
    cG = cnx.from_networkx(G, edge_attrs={"y": 0})
    cp.testing.assert_array_equal(cG.row_indices, [0, 1])
    cp.testing.assert_array_equal(cG.col_indices, [1, 0])
    cp.testing.assert_array_equal(cG.edge_values["y"], [0, 0])
    assert len(cG.edge_values) == 1
    assert cG.edge_masks == cG.node_values == cG.node_masks == {}
    H = cnx.to_networkx(cG)
    assert list(H.edges(data=True)) == [(0, 1, {"y": 0})]

    # If attribute is completely missing (and no default), then just ignore it
    cG = cnx.from_networkx(G, edge_attrs={"y": None})
    cp.testing.assert_array_equal(cG.row_indices, [0, 1])
    cp.testing.assert_array_equal(cG.col_indices, [1, 0])
    assert sorted(cG.edge_values) == sorted(cG.edge_masks) == []
    H = cnx.to_networkx(cG)
    assert list(H.edges(data=True)) == [(0, 1, {})]

    G.add_edge(0, 2)
    # Some edges are missing 'x' attribute; need to use a mask
    for kwargs in [{"preserve_edge_attrs": True}, {"edge_attrs": {"x": None}}]:
        cG = cnx.from_networkx(G, **kwargs)
        cp.testing.assert_array_equal(cG.row_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(cG.col_indices, [1, 2, 0, 0])
        assert sorted(cG.edge_values) == sorted(cG.edge_masks) == ["x"]
        cp.testing.assert_array_equal(cG.edge_masks["x"], [True, False, True, False])
        cp.testing.assert_array_equal(cG.edge_values["x"][cG.edge_masks["x"]], [2, 2])
    H = cnx.to_networkx(cG)
    assert list(H.edges(data=True)) == [(0, 1, {"x": 2}), (0, 2, {})]

    # Now for something more complicated...
    G = nx.Graph()
    G.add_edge(10, 20, x=1)
    G.add_edge(10, 30, x=2, y=1.5)
    G.add_node(10, foo=100)
    G.add_node(20, foo=200, bar=1000)
    G.add_node(30, foo=300)
    # Some edges have masks, some don't
    for kwargs in [
        {"preserve_edge_attrs": True},
        {"preserve_all_attrs": True},
        {"edge_attrs": {"x": None, "y": None}},
        {"edge_attrs": {"x": 0, "y": None}},
        {"edge_attrs": {"x": 0, "y": None}, "edge_dtypes": {"x": int, "y": float}},
    ]:
        cG = cnx.from_networkx(G, **kwargs)
        assert cG.id_to_key == {0: 10, 1: 20, 2: 30}  # Remap node IDs to 0, 1, ...
        cp.testing.assert_array_equal(cG.row_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(cG.col_indices, [1, 2, 0, 0])
        cp.testing.assert_array_equal(cG.edge_values["x"], [1, 2, 1, 2])
        assert sorted(cG.edge_masks) == ["y"]
        cp.testing.assert_array_equal(cG.edge_masks["y"], [False, True, False, True])
        cp.testing.assert_array_equal(
            cG.edge_values["y"][cG.edge_masks["y"]], [1.5, 1.5]
        )
        H = cnx.to_networkx(cG)
        assert G.adj == H.adj

    # Some nodes have masks, some don't
    for kwargs in [
        {"preserve_node_attrs": True},
        {"preserve_all_attrs": True},
        {"node_attrs": {"foo": None, "bar": None}},
        {"node_attrs": {"foo": 0, "bar": None, "missing": None}},
    ]:
        cG = cnx.from_networkx(G, **kwargs)
        assert cG.id_to_key == {0: 10, 1: 20, 2: 30}  # Remap node IDs to 0, 1, ...
        cp.testing.assert_array_equal(cG.row_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(cG.col_indices, [1, 2, 0, 0])
        cp.testing.assert_array_equal(cG.node_values["foo"], [100, 200, 300])
        assert sorted(cG.node_masks) == ["bar"]
        cp.testing.assert_array_equal(cG.node_masks["bar"], [False, True, False])
        cp.testing.assert_array_equal(
            cG.node_values["bar"][cG.node_masks["bar"]], [1000]
        )
        H = cnx.to_networkx(cG)
        assert G.nodes == H.nodes

    # Check default values for nodes
    for kwargs in [
        {"node_attrs": {"foo": None, "bar": 0}},
        {"node_attrs": {"bar": 0}},
        {"node_attrs": {"bar": 0}, "node_dtypes": {"bar": int}},
        {"node_attrs": {"bar": 0, "foo": None}, "node_dtypes": int},
    ]:
        cG = cnx.from_networkx(G, **kwargs)
        assert cG.id_to_key == {0: 10, 1: 20, 2: 30}  # Remap node IDs to 0, 1, ...
        cp.testing.assert_array_equal(cG.row_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(cG.col_indices, [1, 2, 0, 0])
        cp.testing.assert_array_equal(cG.node_values["bar"], [0, 1000, 0])
        assert cG.node_masks == {}

    with pytest.raises(
        TypeError, match="edge_attrs and weight arguments should not both be given"
    ):
        cnx.from_networkx(G, edge_attrs={"x": 1}, weight="x")
    with pytest.raises(TypeError, match="Expected networkx.Graph"):
        cnx.from_networkx({})
