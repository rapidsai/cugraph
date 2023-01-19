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
from cugraph.utilities.utils import import_optional

th = import_optional("torch")


def assert_same_node_feats(gs, g):
    set(gs.ndata.keys()) == set(g.ndata.keys())

    for key in g.ndata.keys():
        for ntype in g.ntypes:
            indices = th.arange(0, g.num_nodes(ntype), dtype=g.idtype).cuda()
            if len(g.ntypes) <= 1 or ntype in g.ndata[key]:
                g_output = g.get_node_storage(key=key, ntype=ntype).fetch(
                    indices, device="cuda"
                )
                gs_output = gs.get_node_storage(key=key, ntype=ntype).fetch(indices)
                equal_t = (gs_output != g_output).sum().cpu()
                assert equal_t == 0


def assert_same_num_nodes(gs, g):
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == gs.num_nodes(ntype)


def assert_same_num_edges_can_etypes(gs, g):
    for can_etype in g.canonical_etypes:
        assert g.num_edges(can_etype) == gs.num_edges(can_etype)


def assert_same_num_edges_etypes(gs, g):
    for etype in g.etypes:
        assert g.num_edges(etype) == gs.num_edges(etype)


def assert_same_edge_feats(gs, g):
    set(gs.edata.keys()) == set(g.edata.keys())
    for key in g.edata.keys():
        for etype in g.canonical_etypes:
            indices = th.arange(0, g.num_edges(etype), dtype=g.idtype).cuda()
            if len(g.etypes) <= 1 or etype in g.edata[key]:
                g_output = g.get_edge_storage(key=key, etype=etype).fetch(
                    indices, device="cuda"
                )
                gs_output = gs.get_edge_storage(key=key, etype=etype).fetch(indices)
                equal_t = (gs_output != g_output).sum().cpu()
                assert equal_t == 0


def assert_same_sampling_len(dgl_g, cugraph_gs, nodes, fanout, edge_dir):
    dgl_o = dgl_g.sample_neighbors(nodes, fanout=fanout, edge_dir=edge_dir)
    cugraph_o = cugraph_gs.sample_neighbors(nodes, fanout=fanout, edge_dir=edge_dir)
    assert cugraph_o.num_edges() == dgl_o.num_edges()
    for etype in dgl_o.canonical_etypes:
        assert dgl_o.num_edges(etype) == cugraph_o.num_edges(etype)
