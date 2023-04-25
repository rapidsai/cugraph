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
from random import randint

from cugraph.utilities.utils import import_optional, MissingModule

import pytest


torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
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


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
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


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("edge_index_type", ["numpy", "torch-cpu", "torch-gpu", "cudf"])
def test_get_edge_index(graph, edge_index_type):
    F, G, N = graph
    if "torch" in edge_index_type:
        if edge_index_type == "torch-cpu":
            device = "cpu"
        else:
            device = "cuda"
        for et in list(G.keys()):
            G[et][0] = torch.as_tensor(G[et][0], device=device)
            G[et][1] = torch.as_tensor(G[et][1], device=device)
    elif edge_index_type == "cudf":
        for et in list(G.keys()):
            G[et][0] = cudf.Series(G[et][0])
            G[et][1] = cudf.Series(G[et][1])

    cugraph_store = CuGraphStore(F, G, N)

    for pyg_can_edge_type in G:
        src, dst = cugraph_store.get_edge_index(
            edge_type=pyg_can_edge_type, layout="coo", is_sorted=False
        )

        if edge_index_type == "cudf":
            assert G[pyg_can_edge_type][0].values_host.tolist() == src.tolist()
            assert G[pyg_can_edge_type][1].values_host.tolist() == dst.tolist()
        else:
            assert G[pyg_can_edge_type][0].tolist() == src.tolist()
            assert G[pyg_can_edge_type][1].tolist() == dst.tolist()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_edge_types(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    eta = cugraph_store._edge_types_to_attrs
    assert eta.keys() == G.keys()

    for attr_name, attr_repr in eta.items():
        assert len(G[attr_name][0]) == attr_repr.size[-1]
        assert attr_name == attr_repr.edge_type


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_get_subgraph(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    if len(G.keys()) > 1:
        for edge_type in G.keys():
            # Subgraphing is not implemented yet and should raise an error
            with pytest.raises(ValueError):
                sg = cugraph_store._subgraph([edge_type])

    sg = cugraph_store._subgraph(list(G.keys()))
    assert isinstance(sg, cugraph.MultiGraph)

    num_edges = sum([len(v[0]) for v in G.values()])
    assert sg.number_of_edges() == num_edges


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_renumber_vertices_basic(single_vertex_graph):
    F, G, N = single_vertex_graph
    cugraph_store = CuGraphStore(F, G, N)

    nodes_of_interest = torch.as_tensor(
        cupy.random.randint(0, sum(N.values()), 3), device="cuda"
    )

    index = cugraph_store._get_vertex_groups_from_sample(nodes_of_interest)
    assert index["vt1"].tolist() == sorted(nodes_of_interest.tolist())


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_renumber_vertices_multi_edge_multi_vertex(multi_edge_multi_vertex_graph_1):
    F, G, N = multi_edge_multi_vertex_graph_1
    cugraph_store = CuGraphStore(F, G, N)

    nodes_of_interest = torch.as_tensor(
        cupy.random.randint(0, sum(N.values()), 3), device="cuda"
    ).unique()

    index = cugraph_store._get_vertex_groups_from_sample(nodes_of_interest)

    black_nodes = nodes_of_interest[nodes_of_interest <= 1]
    brown_nodes = nodes_of_interest[nodes_of_interest > 1] - 2

    if len(black_nodes) > 0:
        assert index["black"].tolist() == sorted(black_nodes.tolist())
    if len(brown_nodes) > 0:
        assert index["brown"].tolist() == sorted(brown_nodes.tolist())


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_renumber_edges(graph):
    """
    FIXME this test is not very good and should be replaced,
    probably with a test that uses known good values.
    """

    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    v_offsets = [N[v] for v in sorted(N.keys())]
    v_offsets = np.array(v_offsets)

    cumsum = v_offsets.cumsum(0)
    v_offsets = cumsum - v_offsets
    v_offsets = {k: int(v_offsets[i]) for i, k in enumerate(sorted(N.keys()))}

    e_num = {
        pyg_can_edge_type: i for i, pyg_can_edge_type in enumerate(sorted(G.keys()))
    }

    eoi_src = np.array([], dtype="int64")
    eoi_dst = np.array([], dtype="int64")
    eoi_type = np.array([], dtype="int32")
    for pyg_can_edge_type, ei in G.items():
        src_type, _, dst_type = pyg_can_edge_type

        c = randint(0, len(ei[0]))  # number to select
        sel = np.random.randint(0, len(ei[0]), c)

        src_i = np.array(ei[0][sel]) + v_offsets[src_type]
        dst_i = np.array(ei[1][sel]) + v_offsets[dst_type]
        eoi_src = np.concatenate([eoi_src, src_i])
        eoi_dst = np.concatenate([eoi_dst, dst_i])
        eoi_type = np.concatenate([eoi_type, np.array([e_num[pyg_can_edge_type]] * c)])

    nodes_of_interest, _ = torch.sort(
        torch.as_tensor(
            np.unique(np.concatenate([eoi_src, eoi_dst])),
        ).cuda()
    )

    noi_index = cugraph_store._get_vertex_groups_from_sample(nodes_of_interest)

    sdf = cudf.DataFrame(
        {
            "sources": eoi_src,
            "destinations": eoi_dst,
            "edge_type": eoi_type,
        }
    ).reset_index(drop=True)

    row, col = cugraph_store._get_renumbered_edge_groups_from_sample(sdf, noi_index)

    for pyg_can_edge_type in G:
        df = cudf.DataFrame(
            {
                "src": cupy.asarray(G[pyg_can_edge_type][0]),
                "dst": cupy.asarray(G[pyg_can_edge_type][1]),
            }
        )

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
            df = df[df.src == src_i]
            df = df[df.dst == dst_i]
            # Ensure only 1 entry matches
            assert len(df) == 1


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_get_tensor(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    for feature_name, feature_on_types in F.get_feature_list().items():
        for type_name in feature_on_types:
            v_ids = np.arange(N[type_name])
            base_series = F.get_data(
                v_ids,
                type_name=type_name,
                feat_name=feature_name,
            ).tolist()

            tsr = cugraph_store.get_tensor(
                type_name, feature_name, v_ids, None, cupy.int64
            ).tolist()

            assert tsr == base_series

@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_get_tensor_empty_idx(karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = CuGraphStore(F, G, N)

    t = cugraph_store.get_tensor(CuGraphTensorAttr(group_name='type0', attr_name='prop0', index=None))
    assert t.tolist() == (torch.arange(17, dtype=torch.float32) * 31).tolist()

@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_multi_get_tensor(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    for vertex_type in sorted(N.keys()):
        v_ids = np.arange(N[vertex_type])
        feat_names = list(F.get_feature_list().keys())
        base_series = None
        for feat_name in feat_names:
            if base_series is None:
                base_series = F.get_data(v_ids, vertex_type, feat_name)
            else:
                base_series = np.stack(
                    [base_series, F.get_data(v_ids, vertex_type, feat_name)]
                )

        tsr = cugraph_store.multi_get_tensor(
            [
                CuGraphTensorAttr(vertex_type, feat_name, v_ids)
                for feat_name in feat_names
            ]
        )

        assert torch.stack(tsr).tolist() == base_series.tolist()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_get_all_tensor_attrs(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    tensor_attrs = []
    for vertex_type in sorted(N.keys()):
        for prop in ["prop1", "prop2"]:
            tensor_attrs.append(
                CuGraphTensorAttr(
                    vertex_type,
                    prop,
                    properties=None,
                    dtype=F.get_data([0], vertex_type, "prop1").dtype,
                )
            )

    for t in tensor_attrs:
        print(t)

    print("\n\n")

    for t in cugraph_store.get_all_tensor_attrs():
        print(t)

    assert sorted(tensor_attrs, key=lambda a: (a.group_name, a.attr_name)) == sorted(
        cugraph_store.get_all_tensor_attrs(), key=lambda a: (a.group_name, a.attr_name)
    )


@pytest.mark.skip("not implemented")
def test_get_tensor_spec_props(graph):
    raise NotImplementedError("not implemented")


@pytest.mark.skip("not implemented")
def test_multi_get_tensor_spec_props(multi_edge_multi_vertex_graph_1):
    raise NotImplementedError("not implemented")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_get_tensor_from_tensor_attrs(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    tensor_attrs = cugraph_store.get_all_tensor_attrs()
    for tensor_attr in tensor_attrs:
        v_ids = np.arange(N[tensor_attr.group_name])
        data = F.get_data(v_ids, tensor_attr.group_name, tensor_attr.attr_name)

        tensor_attr.index = v_ids
        assert cugraph_store.get_tensor(tensor_attr).tolist() == data.tolist()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_get_tensor_size(graph):
    F, G, N = graph
    cugraph_store = CuGraphStore(F, G, N)

    tensor_attrs = cugraph_store.get_all_tensor_attrs()
    for tensor_attr in tensor_attrs:
        sz = N[tensor_attr.group_name]

        tensor_attr.index = np.arange(sz)
        assert cugraph_store.get_tensor_size(tensor_attr) == torch.Size((sz,))

@pytest.mark.skip(reason='pyg bug: https://github.com/pyg-team/pytorch_geometric/issues/7232')
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(torch_geometric, MissingModule), reason="pyg not available")
def test_get_input_nodes(karate_gnn):
    F, G, N = karate_gnn
    cugraph_store = CuGraphStore(F, G, N)

    node_type, input_nodes = torch_geometric.loader.utils.get_input_nodes((cugraph_store, cugraph_store), 'type0')
    
    assert node_type == 'type0'
    assert input_nodes.tolist() == torch.arange(17, dtype=torch.int32).tolist()
    

def test_serialize(multi_edge_multi_vertex_no_graph_1):
    import pickle

    F, G, N = multi_edge_multi_vertex_no_graph_1
    cugraph_store = CuGraphStore(F, G, N)

    cugraph_store_copy = pickle.loads(pickle.dumps(cugraph_store))

    for tensor_attr in cugraph_store.get_all_tensor_attrs():
        sz = cugraph_store.get_tensor_size(tensor_attr)[0]
        tensor_attr.index = np.arange(sz)
        assert cugraph_store.get_tensor(tensor_attr).tolist() == cugraph_store_copy.get_tensor(
            tensor_attr
        ).tolist()

    # Currently does not store edgelist properly for SG
    """
    for edge_attr in cugraph_store.get_all_edge_attrs():
        assert cugraph_store.get_edge_index(edge_attr) \
            == cugraph_store_copy.get_edge_index(edge_attr)
    """
