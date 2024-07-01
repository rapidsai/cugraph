# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

try:
    import cugraph_dgl
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
import cudf
import numpy as np
from cugraph_dgl import CuGraphStorage
from .utils import assert_same_sampling_len

th = import_optional("torch")
dgl = import_optional("dgl")


@pytest.fixture()
def dgl_graph():
    graph_data = {
        ("nt.a", "connects", "nt.b"): (
            th.tensor([0, 1, 2]),
            th.tensor([0, 1, 2]),
        ),
        ("nt.a", "connects", "nt.c"): (
            th.tensor([0, 1, 2]),
            th.tensor([0, 1, 2]),
        ),
        ("nt.c", "connects", "nt.c"): (
            th.tensor([1, 3, 4, 5]),
            th.tensor([0, 0, 0, 0]),
        ),
    }
    g = dgl.heterograph(graph_data)
    return g


def test_cugraphstore_basic_apis():

    num_nodes_dict = {"drug": 3, "gene": 2, "disease": 1}
    # edges
    drug_interacts_drug_df = cudf.DataFrame({"src": [0, 1], "dst": [1, 2]})
    drug_interacts_gene = cudf.DataFrame({"src": [0, 1], "dst": [0, 1]})
    drug_treats_disease = cudf.DataFrame({"src": [1], "dst": [0]})
    data_dict = {
        ("drug", "interacts", "drug"): drug_interacts_drug_df,
        ("drug", "interacts", "gene"): drug_interacts_gene,
        ("drug", "treats", "disease"): drug_treats_disease,
    }
    gs = CuGraphStorage(data_dict=data_dict, num_nodes_dict=num_nodes_dict)
    # add node data
    gs.add_node_data(
        ntype="drug",
        feat_name="node_feat",
        feat_obj=th.as_tensor([0.1, 0.2, 0.3], dtype=th.float64),
    )
    # add edge data
    gs.add_edge_data(
        canonical_etype=("drug", "interacts", "drug"),
        feat_name="edge_feat",
        feat_obj=th.as_tensor([0.2, 0.4], dtype=th.float64),
    )

    assert gs.num_nodes() == 6

    assert gs.num_edges(("drug", "interacts", "drug")) == 2
    assert gs.num_edges(("drug", "interacts", "gene")) == 2
    assert gs.num_edges(("drug", "treats", "disease")) == 1

    node_feat = (
        gs.get_node_storage(key="node_feat", ntype="drug")
        .fetch([0, 1, 2])
        .to("cpu")
        .numpy()
    )
    np.testing.assert_equal(node_feat, np.asarray([0.1, 0.2, 0.3]))

    edge_feat = (
        gs.get_edge_storage(key="edge_feat", etype=("drug", "interacts", "drug"))
        .fetch([0, 1])
        .to("cpu")
        .numpy()
    )
    np.testing.assert_equal(edge_feat, np.asarray([0.2, 0.4]))


def test_sampling_heterograph(dgl_graph):
    cugraph_gs = cugraph_dgl.cugraph_storage_from_heterograph(dgl_graph)

    for fanout in [1, 2, 3, -1]:
        for ntype in ["nt.a", "nt.b", "nt.c"]:
            for d in ["in", "out"]:
                assert_same_sampling_len(
                    dgl_graph,
                    cugraph_gs,
                    nodes={ntype: [0]},
                    fanout=fanout,
                    edge_dir=d,
                )


def test_sampling_homogenous():
    src_ar = np.asarray([0, 1, 2, 0, 1, 2, 7, 9, 10, 11], dtype=np.int32)
    dst_ar = np.asarray([3, 4, 5, 6, 7, 8, 6, 6, 6, 6], dtype=np.int32)
    g = dgl.heterograph({("a", "connects", "a"): (src_ar, dst_ar)})
    cugraph_gs = cugraph_dgl.cugraph_storage_from_heterograph(g)
    # Convert to homogeneous
    g = dgl.to_homogeneous(g)
    nodes = [6]
    # Test for multiple fanouts
    for fanout in [1, 2, 3]:
        exp_g = g.sample_neighbors(nodes, fanout=fanout)
        cu_g = cugraph_gs.sample_neighbors(nodes, fanout=fanout)
        exp_src, exp_dst = exp_g.edges()
        cu_src, cu_dst = cu_g.edges()
        assert len(exp_src) == len(cu_src)

    # Test same results for all neighbours
    exp_g = g.sample_neighbors(nodes, fanout=-1)
    cu_g = cugraph_gs.sample_neighbors(nodes, fanout=-1)
    exp_src, exp_dst = exp_g.edges()
    exp_src, exp_dst = exp_src.numpy(), exp_dst.numpy()

    cu_src, cu_dst = cu_g.edges()
    cu_src, cu_dst = cu_src.to("cpu").numpy(), cu_dst.to("cpu").numpy()

    # Assert same values sorted by src
    exp_src_perm = exp_src.argsort()
    exp_src = exp_src[exp_src_perm]
    exp_dst = exp_dst[exp_src_perm]

    cu_src_perm = cu_src.argsort()
    cu_src = cu_src[cu_src_perm]
    cu_dst = cu_dst[cu_src_perm]

    np.testing.assert_equal(exp_dst, cu_dst)
    np.testing.assert_equal(exp_src, cu_src)
