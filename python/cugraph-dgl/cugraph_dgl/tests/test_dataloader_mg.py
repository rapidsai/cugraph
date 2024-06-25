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

import dgl
import torch as th
from cugraph_dgl import cugraph_storage_from_heterograph
import tempfile
import numpy as np


def sample_dgl_graphs(g, train_nid, fanouts):
    # Single fanout to match cugraph
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=1,
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


def sample_cugraph_dgl_graphs(cugraph_gs, train_nid, fanouts):
    sampler = cugraph_dgl.dataloading.NeighborSampler(fanouts)
    tempdir_object = tempfile.TemporaryDirectory()
    sampling_output_dir = tempdir_object
    dataloader = cugraph_dgl.dataloading.DataLoader(
        cugraph_gs,
        train_nid,
        sampler,
        batch_size=1,
        sampling_output_dir=sampling_output_dir.name,
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


def test_same_heterograph_results(dask_client):
    single_gpu = False
    data_dict = {
        ("B", "BA", "A"): ([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1]),
        ("C", "CA", "A"): ([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1]),
    }
    train_nid = {"A": th.tensor([0])}
    # Create a heterograph with 3 node types and 3 edges types.
    dgl_g = dgl.heterograph(data_dict)
    cugraph_gs = cugraph_storage_from_heterograph(dgl_g, single_gpu=single_gpu)

    dgl_output = sample_dgl_graphs(dgl_g, train_nid, [{"BA": 1, "CA": 1}])
    cugraph_output = sample_cugraph_dgl_graphs(cugraph_gs, train_nid, [2])

    cugraph_output_nodes = cugraph_output[0]["output_nodes"]["A"].cpu().numpy()
    dgl_output_nodes = dgl_output[0]["output_nodes"]["A"].cpu().numpy()
    np.testing.assert_array_equal(cugraph_output_nodes, dgl_output_nodes)
    assert (
        dgl_output[0]["blocks"][0].num_edges()
        == cugraph_output[0]["blocks"][0].num_edges()
    )
    assert (
        dgl_output[0]["blocks"][0].num_dst_nodes()
        == cugraph_output[0]["blocks"][0].num_dst_nodes()
    )


def test_same_homogeneousgraph_results(dask_client):
    single_gpu = False
    train_nid = th.tensor([1])
    # Create a heterograph with 3 node types and 3 edges types.
    dgl_g = dgl.graph(([1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 0, 1, 1, 1, 1]))
    cugraph_gs = cugraph_storage_from_heterograph(dgl_g, single_gpu=single_gpu)

    dgl_output = sample_dgl_graphs(dgl_g, train_nid, [2])
    cugraph_output = sample_cugraph_dgl_graphs(cugraph_gs, train_nid, [2])

    cugraph_output_nodes = cugraph_output[0]["output_nodes"].cpu().numpy()
    dgl_output_nodes = dgl_output[0]["output_nodes"].cpu().numpy()
    np.testing.assert_array_equal(cugraph_output_nodes, dgl_output_nodes)
    assert (
        dgl_output[0]["blocks"][0].num_dst_nodes()
        == cugraph_output[0]["blocks"][0].num_dst_nodes()
    )
    assert (
        dgl_output[0]["blocks"][0].num_edges()
        == cugraph_output[0]["blocks"][0].num_edges()
    )
