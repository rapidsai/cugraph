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

import pytest

import os

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

from cugraph_pyg.data import TensorDictFeatureStore, GraphStore
from cugraph_pyg.loader import NeighborLoader, LinkNeighborLoader

from cugraph.gnn import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
)

os.environ["RAPIDS_NO_INITIALIZE"] = "1"

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


def init_pytorch_worker(rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=rank,
        pool_allocator=False,
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

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id, device=rank)


def run_test_neighbor_loader_mg(rank, uid, world_size, specify_size):
    """
    Basic e2e test that covers loading and sampling.
    """
    init_pytorch_worker(rank, world_size, uid)

    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device="cuda")
    dst = torch.as_tensor(df["dst"], device="cuda")

    ei = torch.stack([dst, src])
    ei = torch.tensor_split(ei.clone(), world_size, axis=1)[rank]

    sz = (34, 34) if specify_size else None
    graph_store = GraphStore(is_multi_gpu=True)
    graph_store.put_edge_index(ei, ("person", "knows", "person"), "coo", False, sz)

    feature_store = TensorDictFeatureStore()
    feature_store["person", "feat"] = torch.randint(128, (34, 16))

    ix_train = torch.tensor_split(torch.arange(34), world_size, axis=0)[rank]

    loader = NeighborLoader(
        (feature_store, graph_store),
        [5, 5],
        input_nodes=ix_train,
    )

    for batch in loader:
        assert isinstance(batch, torch_geometric.data.Data)
        assert (feature_store["person", "feat"][batch.n_id] == batch.feat).all()

    cugraph_comms_shutdown()


@pytest.mark.skip(reason="asdf")
@pytest.mark.parametrize("specify_size", [True, False])
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_neighbor_loader_mg(specify_size):
    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_neighbor_loader_mg,
        args=(
            uid,
            world_size,
            specify_size,
        ),
        nprocs=world_size,
    )


def run_test_neighbor_loader_biased_mg(rank, uid, world_size):
    init_pytorch_worker(rank, world_size, uid)

    eix = torch.stack(
        [
            torch.arange(
                3 * (world_size + rank),
                3 * (world_size + rank + 1),
                dtype=torch.int64,
                device="cuda",
            ),
            torch.arange(3 * rank, 3 * (rank + 1), dtype=torch.int64, device="cuda"),
        ]
    )

    graph_store = GraphStore(is_multi_gpu=True)
    graph_store.put_edge_index(eix, ("person", "knows", "person"), "coo")

    feature_store = TensorDictFeatureStore()
    feature_store["person", "feat"] = torch.randint(128, (6 * world_size, 12))
    feature_store[("person", "knows", "person"), "bias"] = torch.concat(
        [torch.tensor([0, 1, 1], dtype=torch.float32) for _ in range(world_size)]
    )

    loader = NeighborLoader(
        (feature_store, graph_store),
        [1],
        input_nodes=torch.arange(
            3 * rank, 3 * (rank + 1), dtype=torch.int64, device="cuda"
        ),
        batch_size=3,
        weight_attr="bias",
    )

    out = list(iter(loader))
    assert len(out) == 1
    out = out[0]

    assert (
        out.edge_index.cpu()
        == torch.tensor(
            [
                [3, 4],
                [1, 2],
            ]
        )
    ).all()

    cugraph_comms_shutdown()


@pytest.mark.skip(reason="asdf")
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_neighbor_loader_biased_mg():
    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_neighbor_loader_biased_mg,
        args=(
            uid,
            world_size,
        ),
        nprocs=world_size,
    )


def run_test_link_neighbor_loader_basic_mg(
    rank,
    uid,
    world_size,
    num_nodes: int,
    num_edges: int,
    select_edges: int,
    batch_size: int,
    num_neighbors: int,
    depth: int,
):
    init_pytorch_worker(rank, world_size, uid)

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = TensorDictFeatureStore()

    eix = torch.randperm(num_edges)[:select_edges]
    graph_store[("n", "e", "n"), "coo"] = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )

    elx = graph_store[("n", "e", "n"), "coo"][:, eix]
    loader = LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[num_neighbors] * depth,
        edge_label_index=elx,
        batch_size=batch_size,
        shuffle=False,
    )

    elx = torch.tensor_split(elx, eix.numel() // batch_size, dim=1)
    for i, batch in enumerate(loader):
        assert (
            batch.input_id.cpu() == torch.arange(i * batch_size, (i + 1) * batch_size)
        ).all()
        assert (elx[i] == batch.n_id[batch.edge_label_index.cpu()]).all()

    cugraph_comms_shutdown()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
@pytest.mark.parametrize("select_edges", [64, 128])
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("depth", [1, 3])
def test_link_neighbor_loader_basic_mg(select_edges, batch_size, depth):
    num_nodes = 25
    num_edges = 128
    num_neighbors = 2

    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_link_neighbor_loader_basic_mg,
        args=(
            uid,
            world_size,
            num_nodes,
            num_edges,
            select_edges,
            batch_size,
            num_neighbors,
            depth,
        ),
        nprocs=world_size,
    )


def run_test_link_neighbor_loader_uneven_mg(rank, uid, world_size, edge_index):
    init_pytorch_worker(rank, world_size, uid)

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = TensorDictFeatureStore()

    batch_size = 1
    graph_store[("n", "e", "n"), "coo"] = torch.tensor_split(
        edge_index, world_size, dim=-1
    )[rank]

    elx = graph_store[("n", "e", "n"), "coo"]  # select all edges on each worker
    loader = LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[2, 2, 2],
        edge_label_index=elx,
        batch_size=batch_size,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        assert (
            batch.input_id.cpu() == torch.arange(i * batch_size, (i + 1) * batch_size)
        ).all()

        assert (elx[:, [i]] == batch.n_id[batch.edge_label_index.cpu()]).all()

    cugraph_comms_shutdown()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_link_neighbor_loader_uneven_mg():
    edge_index = torch.tensor(
        [
            [0, 1, 3, 4, 7],
            [1, 0, 8, 9, 12],
        ]
    )

    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    torch.multiprocessing.spawn(
        run_test_link_neighbor_loader_uneven_mg,
        args=(
            uid,
            world_size,
            edge_index,
        ),
        nprocs=world_size,
    )
