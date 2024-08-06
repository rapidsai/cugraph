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

import numpy as np

import cugraph_dgl

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

from cugraph.gnn import (
    cugraph_comms_create_unique_id,
    cugraph_comms_shutdown,
)

from cugraph_dgl.tests.utils import init_pytorch_worker

torch = import_optional("torch")
dgl = import_optional("dgl")


def run_test_dataloader_basic_homogeneous(rank, world_size, uid):
    init_pytorch_worker(rank, world_size, uid)

    graph = cugraph_dgl.Graph(is_multi_gpu=True)

    num_nodes = karate.number_of_nodes()
    graph.add_nodes(
        num_nodes,
    )

    edf = karate.get_edgelist()
    graph.add_edges(
        u=torch.tensor_split(torch.as_tensor(edf["src"], device="cuda"), world_size)[
            rank
        ],
        v=torch.tensor_split(torch.as_tensor(edf["dst"], device="cuda"), world_size)[
            rank
        ],
    )

    sampler = cugraph_dgl.dataloading.NeighborSampler([5, 5, 5])
    loader = cugraph_dgl.dataloading.FutureDataLoader(
        graph,
        torch.arange(num_nodes),
        sampler,
        batch_size=2,
        use_ddp=True,
    )

    for in_t, out_t, blocks in loader:
        assert len(blocks) == 3
        assert len(out_t) <= 2


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
def test_dataloader_basic_homogeneous():
    uid = cugraph_comms_create_unique_id()
    # Limit the number of GPUs this rest is run with
    world_size = min(torch.cuda.device_count(), 4)

    torch.multiprocessing.spawn(
        run_test_dataloader_basic_homogeneous,
        args=(
            world_size,
            uid,
        ),
        nprocs=world_size,
    )


def sample_dgl_graphs(g, train_nid, fanouts, batch_size=1):
    # Single fanout to match cugraph
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    dgl_output = {}
    for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        dgl_output[batch_id] = {
            "input_nodes": input_nodes,
            "output_nodes": output_nodes,
            "blocks": blocks,
        }
    return dgl_output


def sample_cugraph_dgl_graphs(cugraph_g, train_nid, fanouts, batch_size=1):
    sampler = cugraph_dgl.dataloading.NeighborSampler(fanouts)

    dataloader = cugraph_dgl.dataloading.FutureDataLoader(
        cugraph_g,
        train_nid,
        sampler,
        batch_size=batch_size,
        drop_last=False,
        shuffle=False,
    )

    cugraph_dgl_output = {}
    for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        cugraph_dgl_output[batch_id] = {
            "input_nodes": input_nodes,
            "output_nodes": output_nodes,
            "blocks": blocks,
        }
    return cugraph_dgl_output


def run_test_same_homogeneousgraph_results(rank, world_size, uid, ix, batch_size):
    init_pytorch_worker(rank, world_size, uid)

    src = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    dst = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    local_src = torch.tensor_split(src, world_size)[rank]
    local_dst = torch.tensor_split(dst, world_size)[rank]

    train_nid = torch.tensor(ix)
    # Create a heterograph with 3 node types and 3 edges types.
    dgl_g = dgl.graph((src, dst))

    cugraph_g = cugraph_dgl.Graph(is_multi_gpu=True)
    cugraph_g.add_nodes(9)
    cugraph_g.add_edges(u=local_src, v=local_dst)

    dgl_output = sample_dgl_graphs(dgl_g, train_nid, [2], batch_size=batch_size)
    cugraph_output = sample_cugraph_dgl_graphs(cugraph_g, train_nid, [2], batch_size)

    cugraph_output_nodes = cugraph_output[0]["output_nodes"].cpu().numpy()
    dgl_output_nodes = dgl_output[0]["output_nodes"].cpu().numpy()

    np.testing.assert_array_equal(
        np.sort(cugraph_output_nodes), np.sort(dgl_output_nodes)
    )
    assert (
        dgl_output[0]["blocks"][0].num_dst_nodes()
        == cugraph_output[0]["blocks"][0].num_dst_nodes()
    )
    assert (
        dgl_output[0]["blocks"][0].num_edges()
        == cugraph_output[0]["blocks"][0].num_edges()
    )

    cugraph_comms_shutdown()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("ix", [[1], [1, 0]])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_same_homogeneousgraph_results_mg(ix, batch_size):
    uid = cugraph_comms_create_unique_id()
    # Limit the number of GPUs this rest is run with
    world_size = min(torch.cuda.device_count(), 4)

    torch.multiprocessing.spawn(
        run_test_same_homogeneousgraph_results,
        args=(world_size, uid, ix, batch_size),
        nprocs=world_size,
    )
