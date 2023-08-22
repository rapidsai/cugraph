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

import cudf
import cupy as cp
import numpy as np
from cugraph_dgl.dataloading.utils.sampling_helpers import (
    cast_to_tensor,
    _get_renumber_map,
    _split_tensor,
    _get_tensor_d_from_sampled_df,
    create_homogeneous_sampled_graphs_from_dataframe,
    _get_source_destination_range,
    _create_homogeneous_cugraph_dgl_nn_sparse_graph,
)
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")
cugraph_dgl = import_optional("cugraph_dgl")


def test_casting_empty_array():
    ar = cp.zeros(shape=0, dtype=cp.int32)
    ser = cudf.Series(ar)
    output_tensor = cast_to_tensor(ser)
    assert output_tensor.dtype == torch.int32


def get_dummy_sampled_df():
    df = cudf.DataFrame()
    df["sources"] = [0, 0, 1, 0, 0, 1, 0, 0, 2] + [np.nan] * 4
    df["destinations"] = [1, 2, 0, 1, 2, 1, 2, 0, 1] + [np.nan] * 4
    df["batch_id"] = [0, 0, 0, 1, 1, 1, 2, 2, 2] + [np.nan] * 4
    df["hop_id"] = [0, 1, 1, 0, 1, 1, 0, 1, 1] + [np.nan] * 4
    df["map"] = [4, 7, 10, 13, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    df = df.astype("int32")
    df["hop_id"] = df["hop_id"].astype("uint8")
    df["map"] = df["map"].astype("int64")
    return df


def test_get_renumber_map():

    sampled_df = get_dummy_sampled_df()

    df, renumber_map, renumber_map_batch_indices = _get_renumber_map(sampled_df)

    # Ensure that map was dropped
    assert "map" not in df.columns

    expected_map = torch.as_tensor(
        [10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int32, device="cuda"
    )
    assert torch.equal(renumber_map, expected_map)

    expected_batch_indices = torch.as_tensor([3, 6], dtype=torch.int32, device="cuda")
    assert torch.equal(renumber_map_batch_indices, expected_batch_indices)

    # Ensure we dropped the Nans for rows  corresponding to the renumber_map
    assert len(df) == 9

    t_ls = _split_tensor(renumber_map, renumber_map_batch_indices)
    assert torch.equal(
        t_ls[0], torch.as_tensor([10, 11, 12], dtype=torch.int64, device="cuda")
    )
    assert torch.equal(
        t_ls[1], torch.as_tensor([13, 14, 15], dtype=torch.int64, device="cuda")
    )
    assert torch.equal(
        t_ls[2], torch.as_tensor([16, 17, 18], dtype=torch.int64, device="cuda")
    )


def test_get_tensor_d_from_sampled_df():
    df = get_dummy_sampled_df()
    tensor_d = _get_tensor_d_from_sampled_df(df)

    expected_maps = {}
    expected_maps[0] = torch.as_tensor([10, 11, 12], dtype=torch.int64, device="cuda")
    expected_maps[1] = torch.as_tensor([13, 14, 15], dtype=torch.int64, device="cuda")
    expected_maps[2] = torch.as_tensor([16, 17, 18], dtype=torch.int64, device="cuda")

    for batch_id, batch_td in tensor_d.items():
        batch_df = df[df["batch_id"] == batch_id]
        for hop_id, hop_t in batch_td.items():
            if hop_id != "map":
                hop_df = batch_df[batch_df["hop_id"] == hop_id]
                assert torch.equal(hop_t["sources"], cast_to_tensor(hop_df["sources"]))
                assert torch.equal(
                    hop_t["destinations"], cast_to_tensor(hop_df["destinations"])
                )

        assert torch.equal(batch_td["map"], expected_maps[batch_id])


def test_create_homogeneous_sampled_graphs_from_dataframe():
    sampler = dgl.dataloading.MultiLayerNeighborSampler([2, 2])
    g = dgl.graph(([0, 10, 20], [0, 0, 10])).to("cuda")
    dgl_input_nodes, dgl_output_nodes, dgl_blocks = sampler.sample_blocks(
        g, torch.as_tensor([0]).to("cuda")
    )

    # Directions are reversed in dgl
    s1, d1 = dgl_blocks[0].edges()
    s0, d0 = dgl_blocks[1].edges()
    srcs = cp.concatenate([cp.asarray(s0), cp.asarray(s1)])
    dsts = cp.concatenate([cp.asarray(d0), cp.asarray(d1)])

    nids = dgl_blocks[0].srcdata[dgl.NID]
    nids = cp.concatenate(
        [cp.asarray([2]), cp.asarray([len(nids) + 2]), cp.asarray(nids)]
    )

    df = cudf.DataFrame()
    df["sources"] = srcs
    df["destinations"] = dsts
    df["hop_id"] = [0] * len(s0) + [1] * len(s1)
    df["batch_id"] = 0
    df["map"] = nids

    (
        cugraph_input_nodes,
        cugraph_output_nodes,
        cugraph_blocks,
    ) = create_homogeneous_sampled_graphs_from_dataframe(df)[0]

    assert torch.equal(dgl_input_nodes, cugraph_input_nodes)
    assert torch.equal(dgl_output_nodes, cugraph_output_nodes)

    for c_block, d_block in zip(cugraph_blocks, dgl_blocks):
        ce, cd = c_block.edges()
        de, dd = d_block.edges()
        assert torch.equal(ce, de)
        assert torch.equal(cd, dd)


def test_get_source_destination_range():
    df = get_dummy_sampled_df()
    output_d = _get_source_destination_range(df)

    expected_output = {
        (0, 0): {"sources_range": 0, "destinations_range": 1},
        (0, 1): {"sources_range": 1, "destinations_range": 2},
        (1, 0): {"sources_range": 0, "destinations_range": 1},
        (1, 1): {"sources_range": 1, "destinations_range": 2},
        (2, 0): {"sources_range": 0, "destinations_range": 2},
        (2, 1): {"sources_range": 2, "destinations_range": 1},
    }

    assert output_d == expected_output


def test__create_homogeneous_cugraph_dgl_nn_sparse_graph():
    tensor_d = {
        "sources_range": 1,
        "destinations_range": 2,
        "sources": torch.as_tensor([0, 0, 1, 1], dtype=torch.int64, device="cuda"),
        "destinations": torch.as_tensor([0, 0, 1, 2], dtype=torch.int64, device="cuda"),
    }

    seednodes_range = 10
    sparse_graph = _create_homogeneous_cugraph_dgl_nn_sparse_graph(
        tensor_d, seednodes_range
    )
    assert sparse_graph.num_src_nodes() == 2
    assert sparse_graph.num_dst_nodes() == seednodes_range + 1
    assert isinstance(sparse_graph, cugraph_dgl.nn.SparseGraph)
