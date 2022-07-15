# Copyright (c) 2022, NVIDIA CORPORATION.
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

from collections import defaultdict
import pytest
import cugraph
from cugraph.testing import utils
from cugraph.experimental import PropertyGraph
import numpy as np
import cudf
import cupy as cp
from cugraph.gnn import CuGraphStore


# Test
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_no_graph(graph_file):
    with pytest.raises(TypeError):
        gstore = cugraph.gnn.CuGraphStore()
        gstore.num_edges()


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_using_graph(graph_file):
    with pytest.raises(ValueError):

        cu_M = utils.read_csv_file(graph_file)

        g = cugraph.Graph()
        g.from_cudf_edgelist(
            cu_M, source="0", destination="1", edge_attr="2", renumber=True
        )

        cugraph.gnn.CuGraphStore(graph=g)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_using_pgraph(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(
        cu_M, source="0", destination="1", edge_attr="2", renumber=True
    )

    pG = PropertyGraph()
    pG.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), property_columns=None
    )

    gstore = cugraph.gnn.CuGraphStore(graph=pG)

    assert g.number_of_edges() == pG.num_edges
    assert g.number_of_edges() == gstore.num_edges()
    assert g.number_of_vertices() == pG.num_vertices
    assert g.number_of_vertices() == gstore.num_vertices


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_node_data_pg(graph_file):

    cu_M = utils.read_csv_file(graph_file)

    pG = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pG)
    gstore.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), edge_key="feat"
    )
    edata = gstore.edata["feat"]

    assert edata.shape[0] > 0


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_egonet(graph_file):

    from cugraph.community.egonet import batched_ego_graphs

    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=True)

    pG = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pG, backend_lib='cupy')
    gstore.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), edge_key="edge_feat"
    )

    nodes = [1, 2]

    ego_edge_list1, seeds_offsets1 = gstore.egonet(nodes, k=1)
    ego_edge_list2, seeds_offsets2 = batched_ego_graphs(g, nodes, radius=1)

    assert ego_edge_list1 == ego_edge_list2
    assert seeds_offsets1 == seeds_offsets2


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_workflow(graph_file):
    # from cugraph.community.egonet import batched_ego_graphs

    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=True)

    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), edge_key="feat"
    )
    nodes = gstore.get_vertex_ids()
    num_nodes = len(nodes)

    assert num_nodes > 0

    sampled_nodes = nodes[:5]

    ego_edge_list, seeds_offsets = gstore.egonet(sampled_nodes, k=1)

    assert len(ego_edge_list) > 0


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_sample_neighbors(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=True)

    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(
        cu_M, edge_key="feat", vertex_col_names=("0", "1")
    )

    nodes = gstore.get_vertex_ids()
    num_nodes = len(nodes)

    assert num_nodes > 0

    sampled_nodes = nodes[:5].to_dlpack()

    parents_cap, children_cap, edge_id_cap = gstore.sample_neighbors(
        sampled_nodes, 2
    )

    parents_list = cudf.from_dlpack(parents_cap)
    assert len(parents_list) > 0


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_sample_neighbor_neg_one_fanout(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=True)

    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(
        cu_M, edge_key="edge_k", vertex_col_names=("0", "1")
    )

    nodes = gstore.get_vertex_ids()
    sampled_nodes = nodes[:5].to_dlpack()
    # -1, default fan_out
    parents_cap, children_cap, edge_id_cap = gstore.sample_neighbors(
        sampled_nodes, -1
    )
    parents_list = cudf.from_dlpack(parents_cap)
    assert len(parents_list) > 0


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_n_data(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=True)

    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)

    gstore.add_edge_data(
        cu_M,
        edge_key="feat",
        vertex_col_names=("0", "1"),
    )

    num_nodes = gstore.num_nodes()
    df_feat = cudf.DataFrame()
    df_feat["node_id"] = np.arange(num_nodes)
    df_feat["val0"] = [float(i + 1) for i in range(num_nodes)]
    df_feat["val1"] = [float(i + 2) for i in range(num_nodes)]
    gstore.add_node_data(
        df_feat,
        node_key="node_feat",
        node_col_name="node_id",
    )

    ndata = gstore.ndata["node_feat"]

    assert ndata.shape[0] > 0


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_e_data(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source="0", destination="1", renumber=True)

    pg = PropertyGraph()
    gstore = cugraph.gnn.CuGraphStore(graph=pg)
    gstore.add_edge_data(
        cu_M, vertex_col_names=("0", "1"), edge_key="edge_k"
    )

    edata = gstore.edata["edge_k"]
    assert edata.shape[0] > 0


dataset1 = {
    "merchants": [
        [
            "merchant_id",
            "merchant_locaton",
            "merchant_size",
            "merchant_sales",
            "merchant_num_employees",
        ],
        [
            (11, 78750, 44, 123.2, 12),
            (4, 78757, 112, 234.99, 18),
            (21, 44145, 83, 992.1, 27),
            (16, 47906, 92, 32.43, 5),
            (86, 47906, 192, 2.43, 51),
        ],
    ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [
            (89021, 78757, 0),
            (32431, 78750, 1),
            (89216, 78757, 1),
            (78634, 47906, 0),
        ],
    ],
    "taxpayers": [
        ["payer_id", "amount"],
        [
            (11, 1123.98),
            (4, 3243.7),
            (21, 8932.3),
            (16, 3241.77),
            (86, 789.2),
            (89021, 23.98),
            (78634, 41.77),
        ],
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num"],
        [
            (89021, 11, 33.2, 1639084966.5513437, 123456),
            (89216, 4, None, 1639085163.481217, 8832),
            (78634, 16, 72.0, 1639084912.567394, 4321),
            (32431, 4, 103.2, 1639084721.354346, 98124),
        ],
    ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [
            (89216, 89021, 9),
            (89216, 32431, 9),
            (32431, 78634, 8),
            (78634, 89216, 8),
        ],
    ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [
            (89216, 78634, 11, 5),
            (89021, 89216, 4, 4),
            (89021, 89216, 21, 3),
            (89021, 89216, 11, 3),
            (89021, 78634, 21, 4),
            (78634, 32431, 11, 4),
        ],
    ],
}


# util to create dataframe
def create_df_from_dataset(col_n, rows):
    data_d = defaultdict(list)
    for row in rows:
        for col_id, col_v in enumerate(row):
            data_d[col_n[col_id]].append(col_v)
    return cudf.DataFrame(data_d)


@pytest.fixture()
def dataset1_CuGraphStore():
    """
    Fixture which returns an instance of a CuGraphStore with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """
    merchant_df = create_df_from_dataset(
        dataset1["merchants"][0], dataset1["merchants"][1]
    )
    user_df = create_df_from_dataset(
        dataset1["users"][0], dataset1["users"][1]
    )
    taxpayers_df = create_df_from_dataset(
        dataset1["taxpayers"][0], dataset1["taxpayers"][1]
    )
    transactions_df = create_df_from_dataset(
        dataset1["transactions"][0], dataset1["transactions"][1]
    )
    relationships_df = create_df_from_dataset(
        dataset1["relationships"][0], dataset1["relationships"][1]
    )
    referrals_df = create_df_from_dataset(
        dataset1["referrals"][0], dataset1["referrals"][1]
    )

    pG = PropertyGraph()
    graph = CuGraphStore(pG, backend_lib='cupy')
    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    graph.add_node_data(
        merchant_df, "merchant_id", "merchant_k", "merchant"
    )
    graph.add_node_data(user_df, "user_id", "user_k", "user")
    graph.add_node_data(
        taxpayers_df, "payer_id", "taxpayers_k", "taxpayers"
    )

    graph.add_edge_data(
        referrals_df,
        ("user_id_1", "user_id_2"),
        "referrals_k",
        "referrals",
    )
    graph.add_edge_data(
        relationships_df,
        ("user_id_1", "user_id_2"),
        "relationships_k",
        "relationships",
    )
    graph.add_edge_data(
        transactions_df,
        ("user_id", "merchant_id"),
        "transactions_k",
        "transactions",
    )

    return graph


def test_num_nodes_gs(dataset1_CuGraphStore):
    assert dataset1_CuGraphStore.num_nodes() == 9


def test_num_edges(dataset1_CuGraphStore):
    assert dataset1_CuGraphStore.num_edges() == 14


def test_get_node_storage_gs(dataset1_CuGraphStore):
    fs = dataset1_CuGraphStore.get_node_storage(
        key="merchant_k", ntype="merchant"
    )
    merchent_gs = fs.fetch([11, 4, 21, 316, 11], device="cuda")
    merchant_df = create_df_from_dataset(
        dataset1["merchants"][0], dataset1["merchants"][1]
    )
    cudf_ar = (
        merchant_df.set_index("merchant_id")
        .loc[[11, 4, 21, 316, 11]]
        .values
    )
    assert cp.allclose(cudf_ar, merchent_gs)


def test_get_edge_storage_gs(dataset1_CuGraphStore):
    fs = dataset1_CuGraphStore.get_edge_storage(
        "relationships_k", "relationships"
    )
    relationship_t = fs.fetch([6, 7, 8], device="cuda")

    relationships_df = create_df_from_dataset(
        dataset1["relationships"][0], dataset1["relationships"][1]
    )
    cudf_ar = relationships_df["relationship_type"].iloc[[0, 1, 2]].values

    assert cp.allclose(cudf_ar, relationship_t)


def test_sampling_gs(dataset1_CuGraphStore):
    node_pack = cp.asarray([4]).toDlpack()
    (
        parents_cap,
        children_cap,
        edge_id_cap,
    ) = dataset1_CuGraphStore.sample_neighbors(node_pack, fanout=1)
    x = cudf.from_dlpack(parents_cap)

    assert x is not None
