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
import pytest

try:
    import cugraph_dgl
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from cugraph.utilities.utils import import_optional
from .utils import (
    assert_same_edge_feats,
    assert_same_node_feats,
    assert_same_num_edges_can_etypes,
    assert_same_num_edges_etypes,
    assert_same_num_nodes,
)

th = import_optional("torch")
dgl = import_optional("dgl")
F = import_optional("dgl.backend")


def create_heterograph1(idtype):
    ctx = th.device("cuda")
    graph_data = {
        ("nt.a", "join.1", "nt.a"): (
            F.tensor([0, 1, 2], dtype=idtype),
            F.tensor([0, 1, 2], dtype=idtype),
        ),
        ("nt.a", "join.2", "nt.a"): (
            F.tensor([0, 1, 2], dtype=idtype),
            F.tensor([0, 1, 2], dtype=idtype),
        ),
    }
    g = dgl.heterograph(graph_data, device=th.device("cuda"))
    g.nodes["nt.a"].data["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=ctx)
    return g


def create_heterograph2(idtype):
    ctx = th.device("cuda")

    g = dgl.heterograph(
        {
            ("user", "plays", "game"): (
                F.tensor([0, 1, 1, 2], dtype=idtype),
                F.tensor([0, 0, 1, 1], dtype=idtype),
            ),
            ("developer", "develops", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
            ("developer", "tests", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=th.device("cuda"),
    )

    g.nodes["user"].data["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=ctx)
    g.nodes["user"].data["p"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=ctx)
    g.nodes["game"].data["h"] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=ctx)
    g.nodes["developer"].data["h"] = F.copy_to(F.tensor([3, 3], dtype=idtype), ctx=ctx)
    g.edges["plays"].data["h"] = F.copy_to(
        F.tensor([1, 1, 1, 1], dtype=idtype), ctx=ctx
    )
    return g


def create_heterograph3(idtype):
    ctx = th.device("cuda")

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (
                F.tensor([0, 1, 1, 2, 2, 2], dtype=idtype),
                F.tensor([0, 0, 1, 1, 2, 2], dtype=idtype),
            ),
            ("user", "plays", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=th.device("cuda"),
    )
    g.nodes["user"].data["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=ctx)
    g.nodes["game"].data["h"] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=ctx)
    g.edges["follows"].data["h"] = F.copy_to(
        F.tensor([10, 20, 30, 40, 50, 60], dtype=idtype), ctx=ctx
    )
    g.edges["follows"].data["p"] = F.copy_to(
        F.tensor([1, 2, 3, 4, 5, 6], dtype=idtype), ctx=ctx
    )
    g.edges["plays"].data["h"] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=ctx)
    return g


def create_heterograph4(idtype):
    ctx = th.device("cuda")

    g = dgl.heterograph(
        {
            ("user", "follows", "user"): (
                F.tensor([1, 2], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
            ("user", "plays", "game"): (
                F.tensor([0, 1], dtype=idtype),
                F.tensor([0, 1], dtype=idtype),
            ),
        },
        idtype=idtype,
        device=th.device("cuda"),
    )
    g.nodes["user"].data["h"] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=ctx)
    g.nodes["game"].data["h"] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=ctx)
    g.edges["follows"].data["h"] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=ctx)
    g.edges["plays"].data["h"] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=ctx)
    return g


@pytest.mark.parametrize("idxtype", [th.int32, th.int64])
def test_heterograph_conversion_nodes(idxtype):
    graph_fs = [
        create_heterograph1,
        create_heterograph2,
        create_heterograph3,
        create_heterograph4,
    ]
    for graph_f in graph_fs:
        g = graph_f(idxtype)
        gs = cugraph_dgl.cugraph_storage_from_heterograph(g)

        assert_same_num_nodes(gs, g)
        assert_same_node_feats(gs, g)


@pytest.mark.parametrize("idxtype", [th.int32, th.int64])
def test_heterograph_conversion_edges(idxtype):
    graph_fs = [
        create_heterograph1,
        create_heterograph2,
        create_heterograph3,
        create_heterograph4,
    ]
    for graph_f in graph_fs:
        g = graph_f(idxtype)
        gs = cugraph_dgl.cugraph_storage_from_heterograph(g)

        assert_same_num_edges_can_etypes(gs, g)
        assert_same_num_edges_etypes(gs, g)
        assert_same_edge_feats(gs, g)
