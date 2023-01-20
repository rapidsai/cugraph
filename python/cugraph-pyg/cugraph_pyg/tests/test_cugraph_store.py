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

import cugraph
from cugraph_pyg.data.cugraph_store import (
    CuGraphTensorAttr,
    CuGraphEdgeAttr,
    EdgeLayout,
)
from cugraph_pyg.data import CuGraphStore

import cudf
import cupy
import numpy as np

import pytest

from cugraph.gnn import FeatureStore

from random import randint

@pytest.fixture
def basic_graph_1():
    G = {
        ("vt1", "pig", "vt1"): [
            np.array([0,0,1,2,2,3]),
            np.array([1,2,4,3,4,1])
        ]
    }

    N = {
        "vt1": 5
    }

    F = FeatureStore()
    F.add_data(
        np.array([100, 200, 300, 400, 500]),
        type_name="vt1",
        feat_name="prop1"
    )

    F.add_data(
        np.array([5,4,3,2,1]),
        type_name="vt1",
        feat_name="prop2"
    )

    return F, G, N


@pytest.fixture
def multi_edge_graph_1():
    G = {
        ("vt1", "pig", "vt1"): [
            np.array([0,2,3,1]),
            np.array([1,3,1,4])
        ],
        ("vt1", "dog", "vt1"): [
            np.array([0,3,4]),
            np.array([2,2,3])
        ],
        ("vt1", "cat", "vt1"): [
            np.array([1,2,2]),
            np.array([4,3,4]),
        ]
    }

    N = {
        "vt1": 5
    }

    F = FeatureStore()
    F.add_data(
        np.array([100,200,300,400,500]),
        type_name="vt1",
        feat_name="prop1"
    )

    F.add_data(
        np.array([5,4,3,2,1]),
        type_name="vt1",
        feat_name="prop2"
    )

    return F, G, N


@pytest.fixture
def multi_edge_multi_vertex_graph_1():

    G = {
        ("brown", "horse", "brown"): [
            np.array([0,0]),
            np.array([1,2]),
        ],
        ("brown", "duck", "black"): [
            np.array([1,1,2]),
            np.array([1,0,1]),
        ],
        ("brown", "mongoose", "black"): [
            np.array([2,1]),
            np.array([0,1]),
        ],
        ("black", "cow", "brown"): [
            np.array([0,0]),
            np.array([1,2]),
        ],
        ("black", "snake", "black"): [
            np.array([1]),
            np.array([0]),
        ]
    }

    N = {
        "brown": 3,
        "black": 2
    }

    F = FeatureStore()
    F.add_data(
        np.array([100,200,300]),
        type_name="brown",
        feat_name="prop1"
    )
    
    F.add_data(
        np.array([400,500]),
        type_name="black",
        feat_name="prop1"
    )

    F.add_data(
        np.array([5,4,3]),
        type_name="brown",
        feat_name="prop2"
    )

    F.add_data(
        np.array([2, 1]),
        type_name="black",
        feat_name="prop2"
    )

    return F, G, N


def test_tensor_attr():
    ta = CuGraphTensorAttr("group0", "property1")
    assert not ta.is_fully_specified()
    assert not ta.is_set("index")

    ta.fully_specify()
    assert ta.is_fully_specified()

    other_ta = CuGraphTensorAttr(index=[1, 2, 3])
    ta.update(other_ta)
    assert ta.index == [1, 2, 3]

    casted_ta1 = CuGraphTensorAttr.cast(ta)
    assert casted_ta1 == ta

    casted_ta2 = CuGraphTensorAttr.cast(index=[1, 2, 3])
    assert casted_ta2.index == [1, 2, 3]
    assert not casted_ta2.is_fully_specified()

    casted_ta3 = CuGraphTensorAttr.cast(
        "group2",
        "property2",
        [1, 2, 3],
    )
    assert casted_ta3.group_name == "group2"
    assert casted_ta3.attr_name == "property2"
    assert casted_ta3.index == [1, 2, 3]


def test_edge_attr():
    ea = CuGraphEdgeAttr("type0", EdgeLayout.COO, False, 10)
    assert ea.edge_type == "type0"
    assert ea.layout == EdgeLayout.COO
    assert not ea.is_sorted
    assert ea.size == 10

    ea = CuGraphEdgeAttr(edge_type="type1", layout="csr", is_sorted=True)
    assert ea.size is None

    ea = CuGraphEdgeAttr.cast("type0", EdgeLayout.COO, False, 10)
    assert ea.edge_type == "type0"
    assert ea.layout == EdgeLayout.COO
    assert not ea.is_sorted
    assert ea.size == 10


@pytest.fixture(
    params=[
        "basic_graph_1",
        "multi_edge_graph_1",
        "multi_edge_multi_vertex_graph_1",
    ]
)
def graph(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(params=["basic_graph_1", "multi_edge_graph_1"])
def single_vertex_graph(request):
    return request.getfixturevalue(request.param)


def test_get_edge_index(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N, backend='cupy')

    for pyg_can_edge_type in G:
        src, dst = cugraph_store.get_edge_index(
            edge_type=pyg_can_edge_type,
            layout="coo",
            is_sorted=False
        )

        assert G[pyg_can_edge_type][0].tolist() == src.get().tolist()
        assert G[pyg_can_edge_type][1].tolist() == dst.get().tolist()


def test_edge_types(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N, backend='cupy')

    eta = cugraph_store._edge_types_to_attrs
    assert eta.keys() == G.keys()

    for attr_name, attr_repr in eta.items():
        assert len(G[attr_name][0]) == attr_repr.size[-1]
        assert attr_name == attr_repr.edge_type


def test_get_subgraph(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N, backend='cupy')

    if len(G.keys()) > 1:
        for edge_type in G.keys():
            # Subgraphing is not implemented yet and should raise an error
            with pytest.raises(ValueError):
                sg = cugraph_store._subgraph([edge_type])

    sg = cugraph_store._subgraph(list(G.keys()))
    assert isinstance(sg, cugraph.MultiGraph)

    num_edges = sum([len(v[0]) for v in G.values()])
    assert sg.number_of_edges() == num_edges


def test_renumber_vertices_basic(single_vertex_graph):
    F, G, N = single_vertex_graph
    cugraph_store = CuGraphStore(F, G, N, backend='cupy')

    nodes_of_interest = cudf.from_dlpack(cupy.random.randint(0, sum(N.values()), 3).__dlpack__())

    index = cugraph_store._get_vertex_groups_from_sample(nodes_of_interest)
    assert index['vt1'].get().tolist() == sorted(nodes_of_interest.values_host.tolist())
    

def test_renumber_vertices_multi_edge_multi_vertex(multi_edge_multi_vertex_graph_1):
    F, G, N = multi_edge_multi_vertex_graph_1
    cugraph_store = CuGraphStore(F, G, N, backend='cupy')

    nodes_of_interest = cudf.from_dlpack(cupy.random.randint(0, sum(N.values()), 3).__dlpack__()).unique()

    index = cugraph_store._get_vertex_groups_from_sample(nodes_of_interest)
    
    black_nodes = nodes_of_interest[nodes_of_interest<=1]
    brown_nodes = nodes_of_interest[nodes_of_interest>1] - 2
    
    if len(black_nodes) > 0:
        assert index['black'].get().tolist() == sorted(black_nodes.values_host.tolist())
    if len(brown_nodes) > 0:
        assert index['brown'].get().tolist() == sorted(brown_nodes.values_host.tolist())


def test_renumber_edges(graph):
    """
    FIXME this test is not very good and should be replaced,
    probably with a test that uses known good values.
    """

    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N, backend='cupy')

    v_offsets = [N[v] for v in sorted(N.keys())]
    v_offsets = cupy.array(v_offsets)

    cumsum = v_offsets.cumsum(0)
    v_offsets = cumsum - v_offsets
    v_offsets = {
        k: int(v_offsets[i])
        for i, k in enumerate(sorted(N.keys()))
    }

    e_num = {
        pyg_can_edge_type: i
        for i, pyg_can_edge_type in enumerate(sorted(G.keys()))
    }

    eoi_src = cupy.array([], dtype='int64')
    eoi_dst = cupy.array([], dtype='int64')
    eoi_type = cupy.array([], dtype='int32')
    for pyg_can_edge_type, ei in G.items():
        src_type, _, dst_type = pyg_can_edge_type

        c = randint(0, len(ei[0])) # number to select
        sel = np.random.randint(0, len(ei[0]), c)

        src_i = cupy.array(ei[0][sel]) + v_offsets[src_type]
        dst_i = cupy.array(ei[1][sel]) + v_offsets[dst_type]
        eoi_src = cupy.concatenate([eoi_src, src_i])
        eoi_dst = cupy.concatenate([eoi_dst, dst_i])
        eoi_type = cupy.concatenate([eoi_type, cupy.array([e_num[pyg_can_edge_type]] * c)])
    
    nodes_of_interest = (
        cudf.from_dlpack(cupy.concatenate([eoi_src, eoi_dst]).__dlpack__())
        .unique()
        .sort_values()
    )
    
    noi_index = cugraph_store._get_vertex_groups_from_sample(nodes_of_interest)

    sdf = cudf.DataFrame(
        {
            "sources": eoi_src,
            "destinations": eoi_dst,
            "indices": eoi_type,
        }
    ).reset_index(drop=True)

    row, col = cugraph_store._get_renumbered_edge_groups_from_sample(sdf, noi_index)
    
    for pyg_can_edge_type in G:
        df = cudf.DataFrame({
            'src':G[pyg_can_edge_type][0],
            'dst':G[pyg_can_edge_type][1],
        })

        G[pyg_can_edge_type] = df

    for pyg_can_edge_type in row:
        stype, _, dtype = pyg_can_edge_type
        src = noi_index[stype][row[pyg_can_edge_type]]
        dst = noi_index[dtype][col[pyg_can_edge_type]]
        assert len(src) == len(dst)

        for i in range(len(src)):
            src_i = int(src[i])
            dst_i = int(dst[i])
            
            df = G[pyg_can_edge_type]
            df = df[df.src==src_i]
            df = df[df.dst==dst_i]
            # Ensure only 1 entry matches
            assert len(df) == 1
            
            

def test_get_tensor(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        for property_name in pG.vertex_property_names:
            if property_name != "vertex_type":
                base_series = pG.get_vertex_data(
                    types=[vertex_type], columns=[property_name]
                )

                vertex_ids = base_series[pG.vertex_col_name].to_cupy()
                base_series = base_series[property_name].to_cupy()

                tsr = feature_store.get_tensor(
                    vertex_type, property_name, vertex_ids, [property_name], cupy.int64
                )

                assert list(tsr) == list(base_series)


def test_multi_get_tensor(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        for property_name in pG.vertex_property_names:
            if property_name != "vertex_type":
                base_series = pG.get_vertex_data(
                    types=[vertex_type], columns=[property_name]
                )

                vertex_ids = base_series[pG.vertex_col_name].to_cupy()
                base_series = base_series[property_name].to_cupy()

                tsr = feature_store.multi_get_tensor(
                    [
                        [
                            vertex_type,
                            property_name,
                            vertex_ids,
                            [property_name],
                            cupy.int64,
                        ]
                    ]
                )
                assert len(tsr) == 1
                tsr = tsr[0]

                assert list(tsr) == list(base_series)


def test_get_all_tensor_attrs(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    tensor_attrs = []
    for vertex_type in pG.vertex_types:
        tensor_attrs.append(
            CuGraphTensorAttr(
                vertex_type, "x", properties=["prop1", "prop2"], dtype=cupy.float32
            )
        )

    assert tensor_attrs == list(feature_store.get_all_tensor_attrs())


def test_get_tensor_unspec_props(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    idx = cupy.array([0, 1, 2, 3, 4])

    for vertex_type in pG.vertex_types:
        t = feature_store.get_tensor(vertex_type, "x", idx)

        data = pG.get_vertex_data(
            vertex_ids=cudf.Series(idx), types=vertex_type, columns=["prop1", "prop2"]
        )[["prop1", "prop2"]].to_cupy(dtype=cupy.float32)

        assert t.tolist() == data.tolist()


def test_multi_get_tensor_unspec_props(multi_edge_multi_vertex_graph_1):
    pG = multi_edge_multi_vertex_graph_1
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    idx = cupy.array([0, 1, 2, 3, 4])
    vertex_types = pG.vertex_types

    tensors_to_get = []
    for vertex_type in sorted(vertex_types):
        tensors_to_get.append(CuGraphTensorAttr(vertex_type, "x", idx))

    tensors = feature_store.multi_get_tensor(tensors_to_get)
    assert tensors[0].tolist() == [[400.0, 2.0], [500.0, 1.0]]
    assert tensors[1].tolist() == [[100.0, 5.0], [200.0, 4.0], [300.0, 3.0]]


def test_get_tensor_from_tensor_attrs(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    tensor_attrs = feature_store.get_all_tensor_attrs()
    for tensor_attr in tensor_attrs:
        tensor_attr.index = cupy.array([0, 1, 2, 3, 4])
        data = pG.get_vertex_data(
            vertex_ids=cudf.Series(tensor_attr.index),
            types=tensor_attr.group_name,
            columns=tensor_attr.properties,
        )[tensor_attr.properties].to_cupy(dtype=tensor_attr.dtype)

        assert feature_store.get_tensor(tensor_attr).tolist() == data.tolist()


def test_get_tensor_size(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        for property_name in pG.vertex_property_names:
            if property_name != "vertex_type":
                base_series = pG.get_vertex_data(
                    types=[vertex_type], columns=[property_name]
                )

                vertex_ids = base_series[pG.vertex_col_name].to_cupy()
                size = feature_store.get_tensor_size(
                    vertex_type, property_name, vertex_ids, [property_name], cupy.int64
                )

                assert len(base_series) == size


def test_get_x(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        base_df = pG.get_vertex_data(types=[vertex_type])

        base_x = (
            base_df.drop(pG.vertex_col_name, axis=1)
            .drop(graph_store._old_vertex_col_name, axis=1)
            .drop(pG.type_col_name, axis=1)
            .to_cupy()
            .astype("float32")
        )

        vertex_ids = base_df[pG.vertex_col_name].to_cupy()

        tsr = feature_store.get_tensor(
            vertex_type, "x", vertex_ids, ["prop1", "prop2"], cupy.int64
        )

        for t, b in zip(tsr, base_x):
            assert list(t) == list(b)


def test_get_x_with_pre_renumber(graph):
    pG = graph
    pG.renumber_vertices_by_type()
    feature_store, graph_store = to_pyg(pG, backend="cupy", renumber_graph=False)

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        base_df = pG.get_vertex_data(types=[vertex_type])

        base_x = (
            base_df.drop(pG.vertex_col_name, axis=1)
            .drop(pG.type_col_name, axis=1)
            .to_cupy()
            .astype("float32")
        )

        vertex_ids = base_df[pG.vertex_col_name].to_cupy()

        tsr = feature_store.get_tensor(
            vertex_type, "x", vertex_ids, ["prop1", "prop2"], cupy.int64
        )

        for t, b in zip(tsr, base_x):
            assert list(t) == list(b)


def test_get_x_bad_dtype(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")
    pass


def test_named_tensor(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")
    pass
