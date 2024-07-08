# Copyright (c) 2024, NVIDIA CORPORATION.
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

import os

import pytest

import cugraph_dgl
import pylibcugraph
import cupy
import numpy as np

import cudf

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
    cugraph_comms_get_raft_handle,
)


pylibwholegraph = import_optional("pylibwholegraph")
torch = import_optional("torch")
dgl = import_optional("dgl")


def init_pytorch_worker(rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=rank,
    )

    import cupy

    cupy.cuda.Device(rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    pylibwholegraph.torch.initialize.init(
        rank,
        world_size,
        rank,
        world_size,
    )

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id, device=rank)


def run_test_graph_make_homogeneous_graph_mg(rank, uid, world_size, direction):
    init_pytorch_worker(rank, world_size, uid)

    df = karate.get_edgelist()
    df.src = df.src.astype("int64")
    df.dst = df.dst.astype("int64")
    wgt = np.random.random((len(df),))

    graph = cugraph_dgl.Graph(
        is_multi_gpu=True, ndata_storage="wholegraph", edata_storage="wholegraph"
    )

    # The number of nodes is set globally but features can have
    # any distribution across workers as long as they are in order.
    global_num_nodes = max(df.src.max(), df.dst.max()) + 1
    node_x = np.array_split(np.arange(global_num_nodes, dtype="int64"), world_size)[
        rank
    ]

    # Each worker gets a shuffled, permuted version of the edgelist
    df = df.sample(frac=1.0)
    df.src = (df.src + rank) % global_num_nodes
    df.dst = (df.dst + rank + 1) % global_num_nodes

    graph.add_nodes(global_num_nodes, data={"x": node_x})
    graph.add_edges(df.src, df.dst, {"weight": wgt})
    plc_dgl_graph = graph._graph(direction=direction)

    assert graph.num_nodes() == global_num_nodes
    assert graph.num_edges() == len(df) * world_size
    assert graph.is_homogeneous
    assert graph.is_multi_gpu

    assert (
        graph.nodes()
        == torch.arange(global_num_nodes, dtype=torch.int64, device="cuda")
    ).all()
    ix = torch.arange(len(node_x) * rank, len(node_x) * (rank + 1), dtype=torch.int64)
    assert (graph.nodes[ix]["x"] == torch.as_tensor(node_x, device="cuda")).all()

    assert (
        graph.edges("eid", device="cuda")
        == torch.arange(world_size * len(df), dtype=torch.int64, device="cuda")
    ).all()
    ix = torch.arange(len(df) * rank, len(df) * (rank + 1), dtype=torch.int64)
    assert (graph.edges[ix]["weight"] == torch.as_tensor(wgt, device="cuda")).all()

    plc_handle = pylibcugraph.ResourceHandle(
        cugraph_comms_get_raft_handle().getHandle()
    )

    plc_expected_graph = pylibcugraph.MGGraph(
        plc_handle,
        pylibcugraph.GraphProperties(is_multigraph=True, is_symmetric=False),
        [df.src] if direction == "out" else [df.dst],
        [df.dst] if direction == "out" else [df.src],
        vertices_array=[
            cupy.array_split(cupy.arange(global_num_nodes, dtype="int64"), world_size)[
                rank
            ]
        ],
    )

    # Do the expensive check to make sure this test fails if an invalid
    # graph is constructed.
    v_actual, d_in_actual, d_out_actual = pylibcugraph.degrees(
        plc_handle,
        plc_dgl_graph,
        source_vertices=cupy.arange(global_num_nodes, dtype="int64"),
        do_expensive_check=True,
    )

    v_exp, d_in_exp, d_out_exp = pylibcugraph.degrees(
        plc_handle,
        plc_expected_graph,
        source_vertices=cupy.arange(global_num_nodes, dtype="int64"),
        do_expensive_check=True,
    )

    assert (v_actual == v_exp).all()
    assert (d_in_actual == d_in_exp).all()
    assert (d_out_actual == d_out_exp).all()

    cugraph_comms_shutdown()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_homogeneous_graph_mg(direction):
    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_graph_make_homogeneous_graph_mg,
        args=(
            uid,
            world_size,
            direction,
        ),
        nprocs=world_size,
    )


def run_test_graph_make_heterogeneous_graph_mg(rank, uid, world_size, direction):
    init_pytorch_worker(rank, world_size, uid)

    df = karate.get_edgelist()
    df.src = df.src.astype("int64")
    df.dst = df.dst.astype("int64")

    graph = cugraph_dgl.Graph(is_multi_gpu=True)
    total_num_nodes = max(df.src.max(), df.dst.max()) + 1

    # Each worker gets a shuffled, permuted version of the edgelist
    df = df.sample(frac=1.0)
    df.src = (df.src + rank) % total_num_nodes
    df.dst = (df.dst + rank + 1) % total_num_nodes

    num_nodes_group_1 = total_num_nodes // 2
    num_nodes_group_2 = total_num_nodes - num_nodes_group_1

    node_x_1 = np.array_split(np.random.random((num_nodes_group_1,)), world_size)[rank]
    node_x_2 = np.array_split(np.random.random((num_nodes_group_2,)), world_size)[rank]

    graph.add_nodes(num_nodes_group_1, {"x": node_x_1}, "type1")
    graph.add_nodes(num_nodes_group_2, {"x": node_x_2}, "type2")

    edges_11 = df[(df.src < num_nodes_group_1) & (df.dst < num_nodes_group_1)]
    edges_12 = df[(df.src < num_nodes_group_1) & (df.dst >= num_nodes_group_1)]
    edges_21 = df[(df.src >= num_nodes_group_1) & (df.dst < num_nodes_group_1)]
    edges_22 = df[(df.src >= num_nodes_group_1) & (df.dst >= num_nodes_group_1)]

    edges_12.dst -= num_nodes_group_1
    edges_21.src -= num_nodes_group_1
    edges_22.dst -= num_nodes_group_1
    edges_22.src -= num_nodes_group_1

    total_edges_11 = torch.tensor(len(edges_11), device="cuda", dtype=torch.int64)
    torch.distributed.all_reduce(total_edges_11, torch.distributed.ReduceOp.SUM)
    total_edges_12 = torch.tensor(len(edges_12), device="cuda", dtype=torch.int64)
    torch.distributed.all_reduce(total_edges_12, torch.distributed.ReduceOp.SUM)
    total_edges_21 = torch.tensor(len(edges_21), device="cuda", dtype=torch.int64)
    torch.distributed.all_reduce(total_edges_21, torch.distributed.ReduceOp.SUM)
    total_edges_22 = torch.tensor(len(edges_22), device="cuda", dtype=torch.int64)
    torch.distributed.all_reduce(total_edges_22, torch.distributed.ReduceOp.SUM)

    graph.add_edges(edges_11.src, edges_11.dst, etype=("type1", "e1", "type1"))
    graph.add_edges(edges_12.src, edges_12.dst, etype=("type1", "e2", "type2"))
    graph.add_edges(edges_21.src, edges_21.dst, etype=("type2", "e3", "type1"))
    graph.add_edges(edges_22.src, edges_22.dst, etype=("type2", "e4", "type2"))

    assert not graph.is_homogeneous
    assert graph.is_multi_gpu

    # Verify graph.nodes()
    assert (
        graph.nodes() == torch.arange(total_num_nodes, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.nodes("type1")
        == torch.arange(num_nodes_group_1, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.nodes("type2")
        == torch.arange(num_nodes_group_2, dtype=torch.int64, device="cuda")
    ).all()

    # Verify graph.edges()
    assert (
        graph.edges("eid", etype=("type1", "e1", "type1"))
        == torch.arange(total_edges_11, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.edges("eid", etype=("type1", "e2", "type2"))
        == torch.arange(total_edges_12, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.edges("eid", etype=("type2", "e3", "type1"))
        == torch.arange(total_edges_21, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.edges("eid", etype=("type2", "e4", "type2"))
        == torch.arange(total_edges_22, dtype=torch.int64, device="cuda")
    ).all()

    # Use sampling call to check graph creation
    # This isn't a test of cuGraph sampling with DGL; the options are
    # set to verify the graph only.
    plc_graph = graph._graph(direction)
    assert isinstance(plc_graph, pylibcugraph.MGGraph)
    sampling_output = pylibcugraph.uniform_neighbor_sample(
        graph._resource_handle,
        plc_graph,
        start_list=cupy.arange(total_num_nodes, dtype="int64"),
        batch_id_list=cupy.full(total_num_nodes, rank, dtype="int32"),
        label_list=cupy.arange(world_size, dtype="int32"),
        label_to_output_comm_rank=cupy.arange(world_size, dtype="int32"),
        h_fan_out=np.array([-1], dtype="int32"),
        with_replacement=False,
        do_expensive_check=True,
        with_edge_properties=True,
        prior_sources_behavior="exclude",
        return_dict=True,
    )

    sdf = cudf.DataFrame(
        {
            "majors": sampling_output["majors"],
            "minors": sampling_output["minors"],
            "edge_id": sampling_output["edge_id"],
            "edge_type": sampling_output["edge_type"],
        }
    )

    expected_offsets = {
        0: (0, 0),
        1: (0, num_nodes_group_1),
        2: (num_nodes_group_1, 0),
        3: (num_nodes_group_1, num_nodes_group_1),
    }
    if direction == "in":
        src_col = "minors"
        dst_col = "majors"
    else:
        src_col = "majors"
        dst_col = "minors"

    edges_11["etype"] = 0
    edges_12["etype"] = 1
    edges_21["etype"] = 2
    edges_22["etype"] = 3

    cdf = cudf.concat([edges_11, edges_12, edges_21, edges_22])
    for i in range(len(cdf)):
        row = cdf.iloc[i]
        etype = row["etype"]
        src = row["src"] + expected_offsets[etype][0]
        dst = row["dst"] + expected_offsets[etype][1]

        f = sdf[
            (sdf[src_col] == src) & (sdf[dst_col] == dst) & (sdf["edge_type"] == etype)
        ]
        assert len(f) > 0  # may be multiple, some could be on other GPU

    cugraph_comms_shutdown()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_heterogeneous_graph_mg(direction):
    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_graph_make_heterogeneous_graph_mg,
        args=(
            uid,
            world_size,
            direction,
        ),
        nprocs=world_size,
    )
