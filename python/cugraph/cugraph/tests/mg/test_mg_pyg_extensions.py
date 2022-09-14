# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cugraph.experimental import MGPropertyGraph
from cugraph.gnn.pyg_extensions import to_pyg
from cugraph.gnn.pyg_extensions.data.cugraph_store import (
    CuGraphTensorAttr,
    CuGraphEdgeAttr,
    EdgeLayout
)

import cudf
import dask_cudf
import cupy

import pytest
import re


@pytest.fixture(scope="module")
def basic_property_graph_1(dask_client):
    pG = MGPropertyGraph()
    pG.add_edge_data(
        dask_cudf.from_cudf(
            cudf.DataFrame({
                'src': cupy.array([
                    0,
                    0,
                    1,
                    2,
                    2,
                    3
                ], dtype='int32'),
                'dst': cupy.array([
                    1,
                    2,
                    4,
                    3,
                    4,
                    1
                ], dtype='int32')
            }),
            npartitions=2
        ),
        vertex_col_names=['src', 'dst']
    )

    pG.add_vertex_data(
        dask_cudf.from_cudf(
            cudf.DataFrame({
                'prop1': [
                    100,
                    200,
                    300,
                    400,
                    500
                ],
                'prop2': [
                    5,
                    4,
                    3,
                    2,
                    1
                ],
                'id': cupy.array([
                    0,
                    1,
                    2,
                    3,
                    4
                ], dtype='int32')
            }),
            npartitions=2
        ),
        vertex_col_name='id'
    )

    return pG


@pytest.fixture(scope="module")
def multi_edge_property_graph_1(dask_client):
    df = dask_cudf.from_cudf(
        cudf.DataFrame({
            'src': cupy.array([
                0,
                0,
                1,
                2,
                2,
                3,
                3,
                1,
                2,
                4
            ], dtype='int32'),
            'dst': cupy.array([
                1,
                2,
                4,
                3,
                3,
                1,
                2,
                4,
                4,
                3
            ], dtype='int32'),
            'edge_type': [
                'pig',
                'dog',
                'cat',
                'pig',
                'cat',
                'pig',
                'dog',
                'pig',
                'cat',
                'dog'
            ]
        }),
        npartitions=2
    )

    pG = MGPropertyGraph()
    for edge_type in df.edge_type.unique().compute().to_pandas():
        pG.add_edge_data(
            df[df.edge_type == edge_type],
            vertex_col_names=['src', 'dst'],
            type_name=edge_type
        )

    pG.add_vertex_data(
        dask_cudf.from_cudf(
            cudf.DataFrame({
                'prop1': [
                    100,
                    200,
                    300,
                    400,
                    500
                ],
                'prop2': [
                    5,
                    4,
                    3,
                    2,
                    1
                ],
                'id': cupy.array([
                    0,
                    1,
                    2,
                    3,
                    4
                ], dtype='int32')
            }),
            npartitions=2
        ),
        vertex_col_name='id'
    )

    return pG


@pytest.fixture(scope="module")
def multi_edge_multi_vertex_property_graph_1(dask_client):
    df = dask_cudf.from_cudf(
        cudf.DataFrame({
            'src': cupy.array([
                0,
                0,
                1,
                2,
                2,
                3,
                3,
                1,
                2,
                4
            ], dtype='int32'),
            'dst': cupy.array([
                1,
                2,
                4,
                3,
                3,
                1,
                2,
                4,
                4,
                3
            ], dtype='int32'),
            'edge_type': [
                'horse',
                'horse',
                'duck',
                'duck',
                'mongoose',
                'cow',
                'cow',
                'mongoose',
                'duck',
                'snake'
            ]
        }),
        npartitions=2
    )

    pG = MGPropertyGraph()
    for edge_type in df.edge_type.compute().unique().to_pandas():
        pG.add_edge_data(
            df[df.edge_type == edge_type],
            vertex_col_names=['src', 'dst'],
            type_name=edge_type
        )

    vdf = dask_cudf.from_cudf(
        cudf.DataFrame({
            'prop1': [
                100,
                200,
                300,
                400,
                500
            ],
            'prop2': [
                5,
                4,
                3,
                2,
                1
            ],
            'id': cupy.array([
                0,
                1,
                2,
                3,
                4
            ], dtype='int32'),
            'vertex_type': [
                'brown',
                'brown',
                'brown',
                'black',
                'black',
            ]
        }),
        npartitions=2
    )

    for vertex_type in vdf.vertex_type.unique().compute().to_pandas():
        vd = vdf[vdf.vertex_type == vertex_type].drop('vertex_type', axis=1)
        pG.add_vertex_data(
            vd,
            vertex_col_name='id',
            type_name=vertex_type
        )

    return pG


def test_tensor_attr():
    ta = CuGraphTensorAttr(
        'group0',
        'property1'
    )
    assert not ta.is_fully_specified()
    assert not ta.is_set('index')

    ta.fully_specify()
    assert ta.is_fully_specified()

    other_ta = CuGraphTensorAttr(
        index=[1, 2, 3]
    )
    ta.update(other_ta)
    assert ta.index == [1, 2, 3]

    casted_ta1 = CuGraphTensorAttr.cast(
        ta
    )
    assert casted_ta1 == ta

    casted_ta2 = CuGraphTensorAttr.cast(
        index=[1, 2, 3]
    )
    assert casted_ta2.index == [1, 2, 3]
    assert not casted_ta2.is_fully_specified()

    casted_ta3 = CuGraphTensorAttr.cast(
        'group2',
        'property2',
        [1, 2, 3],
    )
    assert casted_ta3.group_name == 'group2'
    assert casted_ta3.attr_name == 'property2'
    assert casted_ta3.index == [1, 2, 3]


def test_edge_attr():
    ea = CuGraphEdgeAttr(
        'type0',
        EdgeLayout.COO,
        False,
        10
    )
    assert ea.edge_type == 'type0'
    assert ea.layout == EdgeLayout.COO
    assert not ea.is_sorted
    assert ea.size == 10

    ea = CuGraphEdgeAttr(
        edge_type='type1',
        layout='csr',
        is_sorted=True
    )
    assert ea.size is None

    ea = CuGraphEdgeAttr.cast(
        'type0',
        EdgeLayout.COO,
        False,
        10
    )
    assert ea.edge_type == 'type0'
    assert ea.layout == EdgeLayout.COO
    assert not ea.is_sorted
    assert ea.size == 10


@pytest.fixture(
    params=[
        'basic_property_graph_1',
        'multi_edge_property_graph_1',
        'multi_edge_multi_vertex_property_graph_1'
    ]
)
def graph(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=['basic_property_graph_1', 'multi_edge_property_graph_1']
)
def single_vertex_graph(request):
    return request.getfixturevalue(request.param)


def test_get_edge_index(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    for edge_type in pG.edge_types:
        src, dst = graph_store.get_edge_index(
            edge_type=edge_type,
            layout='coo',
            is_sorted=False
        )

        assert pG.get_num_edges(edge_type) == len(src)
        assert pG.get_num_edges(edge_type) == len(dst)

        edge_data = pG.get_edge_data(
            types=[edge_type],
            columns=[pG.src_col_name, pG.dst_col_name]
        )
        edge_df = cudf.DataFrame({
            'src': src,
            'dst': dst
        })
        edge_df['counter'] = 1

        merged_df = cudf.merge(
            edge_data,
            edge_df,
            left_on=[pG.src_col_name, pG.dst_col_name],
            right_on=['src', 'dst']
        )

        assert merged_df.compute().counter.sum() == len(src)


def test_edge_types(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    eta = graph_store._edge_types_to_attrs
    assert eta.keys() == pG.edge_types

    for attr_name, attr_repr in eta.items():
        assert pG.get_num_edges(attr_name) == attr_repr.size
        assert attr_name == attr_repr.edge_type[1]


def test_get_subgraph(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    for edge_type in pG.edge_types:
        sg = graph_store._subgraph([edge_type])
        assert isinstance(sg, cugraph.Graph)
        assert sg.number_of_edges() == pG.get_num_edges(edge_type)

    sg = graph_store._subgraph(pG.edge_types)
    assert isinstance(sg, cugraph.Graph)

    # duplicate edges are automatically dropped in from_edgelist
    cols = [pG.src_col_name, pG.dst_col_name]
    num_edges = pG.get_edge_data(
        columns=cols
    )[cols].drop_duplicates().compute().shape[0]
    assert sg.number_of_edges() == num_edges


def test_neighbor_sample(single_vertex_graph):
    pG = single_vertex_graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    noi_groups, row_dict, col_dict, _ = graph_store.neighbor_sample(
        index=cupy.array([0, 1, 2, 3, 4], dtype='int32'),
        # FIXME The following line should be num_neighbors=[-1] but
        # there is currently a bug in MG uniform_neighbor_sample.
        # Once this bug is fixed, this line should be changed.
        num_neighbors=[8],
        replace=True,
        directed=True,
        edge_types=[
            v.edge_type
            for v in graph_store._edge_types_to_attrs.values()
        ]
    )

    for node_type, node_ids in noi_groups.items():
        actual_vertex_ids = pG.get_vertex_data(
            types=[node_type],
            columns=[pG.vertex_col_name]
        )[pG.vertex_col_name].compute().to_cupy()

        assert list(node_ids) == list(actual_vertex_ids)

    cols = [pG.src_col_name, pG.dst_col_name, pG.type_col_name]
    combined_df = cudf.DataFrame()
    for edge_type, row in row_dict.items():
        col = col_dict[edge_type]
        df = cudf.DataFrame({pG.src_col_name: row, pG.dst_col_name: col})
        df[pG.type_col_name] = edge_type.replace('__', '')
        combined_df = cudf.concat([combined_df, df])

    base_df = pG.get_edge_data().compute()
    base_df = base_df[cols]
    base_df = base_df.sort_values(cols)
    base_df = base_df.reset_index().drop('index', axis=1)

    numbering = noi_groups['']
    renumber_df = cudf.Series(
        range(len(numbering)),
        index=numbering
    )

    combined_df[pG.src_col_name] = (
        renumber_df.loc[combined_df[pG.src_col_name]].to_cupy()
    )
    combined_df[pG.dst_col_name] = (
        renumber_df.loc[combined_df[pG.dst_col_name]].to_cupy()
    )
    combined_df = combined_df.sort_values(cols)
    combined_df = combined_df.reset_index().drop('index', axis=1)

    assert combined_df.to_arrow().to_pylist() == base_df.to_arrow().to_pylist()


def test_neighbor_sample_multi_vertex(
        multi_edge_multi_vertex_property_graph_1):
    pG = multi_edge_multi_vertex_property_graph_1
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    ex = re.compile(r'[A-z]+__([A-z]+)__[A-z]+')

    noi_groups, row_dict, col_dict, _ = graph_store.neighbor_sample(
        index=cupy.array([0, 1, 2, 3, 4], dtype='int32'),
        # FIXME The following line should be num_neighbors=[-1] but
        # there is currently a bug in MG uniform_neighbor_sample.
        # Once this bug is fixed, this line should be changed.
        num_neighbors=[8],
        replace=True,
        directed=True,
        edge_types=[
            v.edge_type
            for v in graph_store._edge_types_to_attrs.values()
        ]
    )

    for pyg_cpp_edge_type, srcs in row_dict.items():
        cugraph_edge_type = ex.match(pyg_cpp_edge_type).groups()[0]
        num_edges = len(pG.get_edge_data(types=[cugraph_edge_type]).compute())
        assert num_edges == len(srcs)


def test_get_tensor(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        for property_name in pG.vertex_property_names:
            if property_name != 'vertex_type':
                base_series = pG.get_vertex_data(
                    types=[vertex_type],
                    columns=[property_name, pG.vertex_col_name]
                )

                vertex_ids = base_series[pG.vertex_col_name]
                vertex_ids = vertex_ids.compute().to_cupy()

                base_series = base_series[property_name]
                base_series = base_series.compute().to_cupy()

                tsr = feature_store.get_tensor(
                    vertex_type,
                    property_name,
                    vertex_ids
                )

                print(base_series)
                print(tsr)
                assert list(tsr) == list(base_series)


def test_multi_get_tensor(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        for property_name in pG.vertex_property_names:
            if property_name != 'vertex_type':
                base_series = pG.get_vertex_data(
                    types=[vertex_type],
                    columns=[property_name, pG.vertex_col_name]
                )

                vertex_ids = base_series[pG.vertex_col_name]
                vertex_ids = vertex_ids.compute().to_cupy()

                base_series = base_series[property_name]
                base_series = base_series.compute().to_cupy()

                tsr = feature_store.multi_get_tensor(
                    [[vertex_type, property_name, vertex_ids]]
                )
                assert len(tsr) == 1
                tsr = tsr[0]

                print(base_series)
                print(tsr)
                assert list(tsr) == list(base_series)


def test_get_all_tensor_attrs(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    tensor_attrs = []
    for vertex_type in pG.vertex_types:
        tensor_attrs.append(CuGraphTensorAttr(
            vertex_type,
            'x'
        ))

    assert tensor_attrs == feature_store.get_all_tensor_attrs()


def test_get_tensor_size(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        for property_name in pG.vertex_property_names:
            if property_name != 'vertex_type':
                base_series = pG.get_vertex_data(
                    types=[vertex_type],
                    columns=[property_name, pG.vertex_col_name]
                )

                vertex_ids = base_series[pG.vertex_col_name]
                vertex_ids = vertex_ids.compute().to_cupy()
                size = feature_store.get_tensor_size(
                    vertex_type,
                    property_name,
                    vertex_ids
                )

                assert len(base_series) == size


def test_get_x(graph):
    pG = graph
    feature_store, graph_store = to_pyg(pG, backend='cupy')

    vertex_types = pG.vertex_types
    for vertex_type in vertex_types:
        base_df = pG.get_vertex_data(
            types=[vertex_type]
        )

        base_x = base_df.drop(
            pG.vertex_col_name, axis=1
        ).drop(
            pG.type_col_name, axis=1
        ).compute().to_cupy().astype('float32')

        vertex_ids = base_df[pG.vertex_col_name].compute().to_cupy()

        tsr = feature_store.get_tensor(
            vertex_type,
            'x',
            vertex_ids
        )

        print(base_x)
        print(tsr)
        for t, b in zip(tsr, base_x):
            assert list(t) == list(b)
