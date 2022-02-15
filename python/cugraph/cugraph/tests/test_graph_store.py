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

import pytest

import cugraph
from cugraph.tests import utils
from cugraph.experimental import PropertyGraph


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
        g.from_cudf_edgelist(cu_M, source='0',
                             destination='1', edge_attr='2', renumber=True)

        cugraph.gnn.CuGraphStore(graph=g)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_using_pgraph(graph_file):
    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source='0', destination='1',
                         edge_attr='2', renumber=True)

    pG = PropertyGraph()
    pG.add_edge_data(cu_M,
                     type_name="edge",
                     vertex_col_names=("0", "1"),
                     property_columns=None)

    gstore = cugraph.gnn.CuGraphStore(graph=pG)

    assert g.number_of_edges() == pG.num_edges
    assert g.number_of_edges() == gstore.num_edges
    assert g.number_of_vertices() == pG.num_vertices
    assert g.number_of_vertices() == gstore.num_vertices


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_node_data_pg(graph_file):
    with pytest.raises(NotImplementedError):

        cu_M = utils.read_csv_file(graph_file)

        pG = PropertyGraph()
        pG.add_edge_data(cu_M,
                         type_name="edge",
                         vertex_col_names=("0", "1"),
                         property_columns=None)

        gstore = cugraph.gnn.CuGraphStore(graph=pG)

        gstore.ndata


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_egonet(graph_file):

    from cugraph.community.egonet import batched_ego_graphs

    cu_M = utils.read_csv_file(graph_file)

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(cu_M, source='0', destination='1', renumber=True)

    pG = PropertyGraph()
    pG.add_edge_data(cu_M,
                     type_name="edge",
                     vertex_col_names=("0", "1"),
                     property_columns=None)

    gstore = cugraph.gnn.CuGraphStore(graph=pG)

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
    g.from_cudf_edgelist(cu_M, source='0', destination='1', renumber=True)

    pg = PropertyGraph()
    pg.add_edge_data(cu_M,
                     type_name="edge",
                     vertex_col_names=("0", "1"),
                     property_columns=["2"])

    gstore = cugraph.gnn.CuGraphStore(graph=pg)

    nodes = gstore.get_vertex_ids()
    num_nodes = len(nodes)

    assert num_nodes > 0

    sampled_nodes = nodes[:5]

    ego_edge_list, seeds_offsets = gstore.egonet(sampled_nodes, k=1)

    assert len(ego_edge_list) > 0
