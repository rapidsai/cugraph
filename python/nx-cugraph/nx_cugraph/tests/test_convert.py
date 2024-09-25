# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import nx_cugraph as nxcg
from nx_cugraph import interface


@pytest.mark.parametrize(
    "graph_class", [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"preserve_edge_attrs": True},
        {"preserve_node_attrs": True},
        {"preserve_all_attrs": True},
        {"edge_attrs": {"x": 0}},
        {"edge_attrs": {"x": None}},
        {"edge_attrs": {"x": nxcg.convert.REQUIRED}},
        {"edge_attrs": {"x": ...}},  # sugar for REQUIRED
        {"edge_attrs": "x"},
        {"node_attrs": {"x": 0}},
        {"node_attrs": {"x": None}},
        {"node_attrs": {"x": nxcg.convert.REQUIRED}},
        {"node_attrs": {"x": ...}},  # sugar for REQUIRED
        {"node_attrs": "x"},
    ],
)
def test_convert_empty(graph_class, kwargs):
    G = graph_class()
    Gcg = nxcg.from_networkx(G, **kwargs)
    H = nxcg.to_networkx(Gcg)
    assert G.number_of_nodes() == Gcg.number_of_nodes() == H.number_of_nodes() == 0
    assert G.number_of_edges() == Gcg.number_of_edges() == H.number_of_edges() == 0
    assert Gcg.edge_values == Gcg.edge_masks == Gcg.node_values == Gcg.node_masks == {}
    assert G.graph == Gcg.graph == H.graph == {}


@pytest.mark.parametrize("graph_class", [nx.Graph, nx.MultiGraph])
def test_convert(graph_class):
    # FIXME: can we break this into smaller tests?
    G = graph_class()
    G.add_edge(0, 1, x=2)
    G.add_node(0, foo=10)
    G.add_node(1, foo=20, bar=100)
    for kwargs in [
        {"preserve_edge_attrs": True},
        {"preserve_all_attrs": True},
        {"edge_attrs": {"x": 0}},
        {"edge_attrs": {"x": None}, "node_attrs": {"bar": None}},
        {"edge_attrs": "x", "edge_dtypes": int},
        {
            "edge_attrs": {"x": nxcg.convert.REQUIRED},
            "node_attrs": {"foo": nxcg.convert.REQUIRED},
        },
        {"edge_attrs": {"x": ...}, "node_attrs": {"foo": ...}},  # sugar for REQUIRED
    ]:
        # All edges have "x" attribute, so all kwargs are equivalent
        Gcg = nxcg.from_networkx(G, **kwargs)
        cp.testing.assert_array_equal(Gcg.src_indices, [0, 1])
        cp.testing.assert_array_equal(Gcg.dst_indices, [1, 0])
        cp.testing.assert_array_equal(Gcg.edge_values["x"], [2, 2])
        assert len(Gcg.edge_values) == 1
        assert Gcg.edge_masks == {}
        H = nxcg.to_networkx(Gcg)
        assert G.number_of_nodes() == Gcg.number_of_nodes() == H.number_of_nodes() == 2
        assert G.number_of_edges() == Gcg.number_of_edges() == H.number_of_edges() == 1
        assert G.adj == H.adj

    with pytest.raises(KeyError, match="bar"):
        nxcg.from_networkx(G, node_attrs={"bar": ...})

    # Structure-only graph (no edge attributes)
    Gcg = nxcg.from_networkx(G, preserve_node_attrs=True)
    cp.testing.assert_array_equal(Gcg.src_indices, [0, 1])
    cp.testing.assert_array_equal(Gcg.dst_indices, [1, 0])
    cp.testing.assert_array_equal(Gcg.node_values["foo"], [10, 20])
    assert Gcg.edge_values == Gcg.edge_masks == {}
    H = nxcg.to_networkx(Gcg)
    if G.is_multigraph():
        assert set(G.edges) == set(H.edges) == {(0, 1, 0)}
    else:
        assert set(G.edges) == set(H.edges) == {(0, 1)}
    assert G.nodes == H.nodes

    # Fill completely missing attribute with default value
    Gcg = nxcg.from_networkx(G, edge_attrs={"y": 0})
    cp.testing.assert_array_equal(Gcg.src_indices, [0, 1])
    cp.testing.assert_array_equal(Gcg.dst_indices, [1, 0])
    cp.testing.assert_array_equal(Gcg.edge_values["y"], [0, 0])
    assert len(Gcg.edge_values) == 1
    assert Gcg.edge_masks == Gcg.node_values == Gcg.node_masks == {}
    H = nxcg.to_networkx(Gcg)
    assert list(H.edges(data=True)) == [(0, 1, {"y": 0})]
    if Gcg.is_multigraph():
        assert set(H.edges) == {(0, 1, 0)}

    # If attribute is completely missing (and no default), then just ignore it
    Gcg = nxcg.from_networkx(G, edge_attrs={"y": None})
    cp.testing.assert_array_equal(Gcg.src_indices, [0, 1])
    cp.testing.assert_array_equal(Gcg.dst_indices, [1, 0])
    assert sorted(Gcg.edge_values) == sorted(Gcg.edge_masks) == []
    H = nxcg.to_networkx(Gcg)
    assert list(H.edges(data=True)) == [(0, 1, {})]
    if Gcg.is_multigraph():
        assert set(H.edges) == {(0, 1, 0)}

    G.add_edge(0, 2)
    # Some edges are missing 'x' attribute; need to use a mask
    for kwargs in [{"preserve_edge_attrs": True}, {"edge_attrs": {"x": None}}]:
        Gcg = nxcg.from_networkx(G, **kwargs)
        cp.testing.assert_array_equal(Gcg.src_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(Gcg.dst_indices, [1, 2, 0, 0])
        assert sorted(Gcg.edge_values) == sorted(Gcg.edge_masks) == ["x"]
        cp.testing.assert_array_equal(Gcg.edge_masks["x"], [True, False, True, False])
        cp.testing.assert_array_equal(Gcg.edge_values["x"][Gcg.edge_masks["x"]], [2, 2])
    H = nxcg.to_networkx(Gcg)
    assert list(H.edges(data=True)) == [(0, 1, {"x": 2}), (0, 2, {})]
    if Gcg.is_multigraph():
        assert set(H.edges) == {(0, 1, 0), (0, 2, 0)}

    with pytest.raises(KeyError, match="x"):
        nxcg.from_networkx(G, edge_attrs={"x": nxcg.convert.REQUIRED})
    with pytest.raises(KeyError, match="x"):
        nxcg.from_networkx(G, edge_attrs={"x": ...})
    with pytest.raises(KeyError, match="bar"):
        nxcg.from_networkx(G, node_attrs={"bar": nxcg.convert.REQUIRED})
    with pytest.raises(KeyError, match="bar"):
        nxcg.from_networkx(G, node_attrs={"bar": ...})

    # Now for something more complicated...
    G = graph_class()
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
        {"edge_attrs": {"x": 0, "y": None}},
        {"edge_attrs": {"x": 0, "y": None}, "edge_dtypes": {"x": int, "y": float}},
    ]:
        Gcg = nxcg.from_networkx(G, **kwargs)
        assert Gcg.id_to_key == [10, 20, 30]  # Remap node IDs to 0, 1, ...
        cp.testing.assert_array_equal(Gcg.src_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(Gcg.dst_indices, [1, 2, 0, 0])
        cp.testing.assert_array_equal(Gcg.edge_values["x"], [1, 2, 1, 2])
        assert sorted(Gcg.edge_masks) == ["y"]
        cp.testing.assert_array_equal(Gcg.edge_masks["y"], [False, True, False, True])
        cp.testing.assert_array_equal(
            Gcg.edge_values["y"][Gcg.edge_masks["y"]], [1.5, 1.5]
        )
        H = nxcg.to_networkx(Gcg)
        assert G.adj == H.adj

    # Some nodes have masks, some don't
    for kwargs in [
        {"preserve_node_attrs": True},
        {"preserve_all_attrs": True},
        {"node_attrs": {"foo": None, "bar": None}},
        {"node_attrs": {"foo": None, "bar": None}},
        {"node_attrs": {"foo": 0, "bar": None, "missing": None}},
    ]:
        Gcg = nxcg.from_networkx(G, **kwargs)
        assert Gcg.id_to_key == [10, 20, 30]  # Remap node IDs to 0, 1, ...
        cp.testing.assert_array_equal(Gcg.src_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(Gcg.dst_indices, [1, 2, 0, 0])
        cp.testing.assert_array_equal(Gcg.node_values["foo"], [100, 200, 300])
        assert sorted(Gcg.node_masks) == ["bar"]
        cp.testing.assert_array_equal(Gcg.node_masks["bar"], [False, True, False])
        cp.testing.assert_array_equal(
            Gcg.node_values["bar"][Gcg.node_masks["bar"]], [1000]
        )
        H = nxcg.to_networkx(Gcg)
        assert G.nodes == H.nodes

    # Check default values for nodes
    for kwargs in [
        {"node_attrs": {"foo": None, "bar": 0}},
        {"node_attrs": {"foo": None, "bar": 0, "missing": None}},
        {"node_attrs": {"bar": 0}},
        {"node_attrs": {"bar": 0}, "node_dtypes": {"bar": int}},
        {"node_attrs": {"bar": 0, "foo": None}, "node_dtypes": int},
    ]:
        Gcg = nxcg.from_networkx(G, **kwargs)
        assert Gcg.id_to_key == [10, 20, 30]  # Remap node IDs to 0, 1, ...
        cp.testing.assert_array_equal(Gcg.src_indices, [0, 0, 1, 2])
        cp.testing.assert_array_equal(Gcg.dst_indices, [1, 2, 0, 0])
        cp.testing.assert_array_equal(Gcg.node_values["bar"], [0, 1000, 0])
        assert Gcg.node_masks == {}

    with pytest.raises(
        TypeError, match="edge_attrs and weight arguments should not both be given"
    ):
        interface.BackendInterface.convert_from_nx(G, edge_attrs={"x": 1}, weight="x")
    with pytest.raises(TypeError, match="Expected networkx.Graph"):
        nxcg.from_networkx({})


@pytest.mark.parametrize("graph_class", [nx.MultiGraph, nx.MultiDiGraph])
def test_multigraph(graph_class):
    G = graph_class()
    G.add_edge(0, 1, "key1", x=10)
    G.add_edge(0, 1, "key2", y=20)
    Gcg = nxcg.from_networkx(G, preserve_edge_attrs=True)
    H = nxcg.to_networkx(Gcg)
    assert type(G) is type(H)
    assert nx.utils.graphs_equal(G, H)


def test_to_dict_of_lists():
    G = nx.MultiGraph()
    G.add_edge("a", "b")
    G.add_edge("a", "c")
    G.add_edge("a", "b")
    expected = nx.to_dict_of_lists(G)
    result = nxcg.to_dict_of_lists(G)
    assert expected == result
    expected = nx.to_dict_of_lists(G, nodelist=["a", "b"])
    result = nxcg.to_dict_of_lists(G, nodelist=["a", "b"])
    assert expected == result
    with pytest.raises(nx.NetworkXError, match="The node d is not in the graph"):
        nx.to_dict_of_lists(G, nodelist=["a", "d"])
    with pytest.raises(nx.NetworkXError, match="The node d is not in the graph"):
        nxcg.to_dict_of_lists(G, nodelist=["a", "d"])
    G.add_node("d")  # No edges
    expected = nx.to_dict_of_lists(G)
    result = nxcg.to_dict_of_lists(G)
    assert expected == result
    expected = nx.to_dict_of_lists(G, nodelist=["a", "d"])
    result = nxcg.to_dict_of_lists(G, nodelist=["a", "d"])
    assert expected == result
    # Now try with default node ids
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 2)
    expected = nx.to_dict_of_lists(G)
    result = nxcg.to_dict_of_lists(G)
    assert expected == result
    expected = nx.to_dict_of_lists(G, nodelist=[0, 1])
    result = nxcg.to_dict_of_lists(G, nodelist=[0, 1])
    assert expected == result
    with pytest.raises(nx.NetworkXError, match="The node 3 is not in the digraph"):
        nx.to_dict_of_lists(G, nodelist=[0, 3])
    with pytest.raises(nx.NetworkXError, match="The node 3 is not in the digraph"):
        nxcg.to_dict_of_lists(G, nodelist=[0, 3])
    G.add_node(3)  # No edges
    expected = nx.to_dict_of_lists(G)
    result = nxcg.to_dict_of_lists(G)
    assert expected == result
    expected = nx.to_dict_of_lists(G, nodelist=[0, 3])
    result = nxcg.to_dict_of_lists(G, nodelist=[0, 3])
    assert expected == result
