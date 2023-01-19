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
        print(pyg_can_edge_type)
        src, dst = cugraph_store.get_edge_index(
            edge_type=pyg_can_edge_type,
            layout="coo",
            is_sorted=False
        )

        assert G[pyg_can_edge_type][0].tolist() == src.get().tolist()
        assert G[pyg_can_edge_type][1].tolist() == dst.get().tolist()

        # check actual values
        print(src,dst)


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


def test_renumber_vertices(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")

    nodes_of_interest = pG.get_vertices().sample(3)
    vc_actual = pG.get_vertex_data(nodes_of_interest)[pG.type_col_name].value_counts()
    index = graph_store._get_vertex_groups_from_sample(nodes_of_interest)

    for vtype in index:
        assert len(index[vtype]) == vc_actual[vtype]


def test_renumber_edges(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend="cupy")
    eoi_df = pG.get_edge_data().sample(4)
    nodes_of_interest = (
        cudf.concat([eoi_df[pG.src_col_name], eoi_df[pG.dst_col_name]])
        .unique()
        .sort_values()
    )
    vd = pG.get_vertex_data(nodes_of_interest)
    noi_index = {
        vd[pG.type_col_name]
        .cat.categories[gg[0]]: vd.loc[gg[1].values_host][pG.vertex_col_name]
        .to_cupy()
        for gg in vd.groupby(pG.type_col_name).groups.items()
    }

    sdf = cudf.DataFrame(
        {
            "sources": eoi_df[pG.src_col_name],
            "destinations": eoi_df[pG.dst_col_name],
            "indices": eoi_df[pG.type_col_name].cat.codes,
        }
    ).reset_index(drop=True)

    row, col = graph_store._get_renumbered_edge_groups_from_sample(sdf, noi_index)

    for etype in row:
        stype, ctype, dtype = etype
        src = noi_index[stype][row[etype]]
        dst = noi_index[dtype][col[etype]]
        assert len(src) == len(dst)

        for i in range(len(src)):
            src_i = int(src[i])
            dst_i = int(dst[i])
            f = eoi_df[eoi_df[pG.src_col_name] == src_i]
            f = f[f[pG.dst_col_name] == dst_i]
            f = f[f[pG.type_col_name] == ctype]
            assert len(f) == 1  # make sure we match exactly 1 edge


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
