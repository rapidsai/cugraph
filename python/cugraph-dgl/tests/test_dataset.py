# Copyright (c) 2023, NVIDIA CORPORATION.
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

    del cugraph_dgl
except ModuleNotFoundError:
    pytest.skip("cugraph_dgl not available", allow_module_level=True)

from dgl.dataloading import MultiLayerNeighborSampler
import dgl
import torch
import cudf
import pandas as pd
import cupy as cp
import numpy as np
from cugraph_dgl.dataloading.utils.sampling_helpers import (
    create_homogeneous_sampled_graphs_from_dataframe,
)


def get_edge_df_from_homogenous_block(block):
    block = block.to("cpu")
    src, dst, eid = block.edges("all")
    src = block.srcdata[dgl.NID][src]
    dst = block.dstdata[dgl.NID][dst]
    eid = block.edata[dgl.EID][eid]
    df = pd.DataFrame({"src": src, "dst": dst, "eid": eid})
    return df.sort_values(by="eid").reset_index(drop=True)


def create_dgl_mfgs(g, seed_nodes, fanout):
    sampler = MultiLayerNeighborSampler(fanout)
    return sampler.sample_blocks(g, seed_nodes)


def create_cugraph_dgl_homogenous_mfgs(dgl_blocks, return_type):
    df_ls = []
    unique_vertices_ls = []
    for hop_id, block in enumerate(reversed(dgl_blocks)):
        block = block.to("cpu")
        src, dst, eid = block.edges("all")
        eid = block.edata[dgl.EID][eid]

        og_src = block.srcdata[dgl.NID][src]
        og_dst = block.dstdata[dgl.NID][dst]
        unique_vertices = pd.concat(
            [pd.Series(og_dst.numpy()), pd.Series(og_src.numpy())]
        ).drop_duplicates(keep="first")
        unique_vertices_ls.append(unique_vertices)
        df = cudf.DataFrame(
            {
                "sources": cp.asarray(src),
                "destinations": cp.asarray(dst),
                "edge_id": cp.asarray(eid),
            }
        )
        df["hop_id"] = hop_id
        df_ls.append(df)
    df = cudf.concat(df_ls, ignore_index=True)
    df["batch_id"] = 0

    # Add map column
    # to the dataframe
    renumberd_map = pd.concat(unique_vertices_ls).drop_duplicates(keep="first").values
    offsets = np.asarray([2, 2 + len(renumberd_map)])
    map_ar = np.concatenate([offsets, renumberd_map])
    map_ser = cudf.Series(map_ar)
    # Have to reindex cause map_ser can be of larger length than df
    df = df.reindex(df.index.union(map_ser.index))
    df["map"] = map_ser
    return create_homogeneous_sampled_graphs_from_dataframe(
        df, return_type=return_type
    )[0]


@pytest.mark.parametrize("return_type", ["dgl.Block", "cugraph_dgl.nn.SparseGraph"])
@pytest.mark.parametrize("seed_node", [3, 4, 5])
def test_homogeneous_sampled_graphs_from_dataframe(return_type, seed_node):
    g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))
    fanout = [1, 1, 1]
    seed_node = torch.as_tensor([seed_node])

    dgl_seed_nodes, dgl_output_nodes, dgl_mfgs = create_dgl_mfgs(g, seed_node, fanout)
    (
        cugraph_seed_nodes,
        cugraph_output_nodes,
        cugraph_mfgs,
    ) = create_cugraph_dgl_homogenous_mfgs(dgl_mfgs, return_type=return_type)

    np.testing.assert_equal(
        cugraph_seed_nodes.cpu().numpy().copy().sort(),
        dgl_seed_nodes.cpu().numpy().copy().sort(),
    )

    np.testing.assert_equal(
        dgl_output_nodes.cpu().numpy().copy().sort(),
        cugraph_output_nodes.cpu().numpy().copy().sort(),
    )

    if return_type == "dgl.Block":
        for dgl_block, cugraph_dgl_block in zip(dgl_mfgs, cugraph_mfgs):
            dgl_df = get_edge_df_from_homogenous_block(dgl_block)
            cugraph_dgl_df = get_edge_df_from_homogenous_block(cugraph_dgl_block)
            pd.testing.assert_frame_equal(dgl_df, cugraph_dgl_df)
    else:
        for dgl_block, cugraph_dgl_graph in zip(dgl_mfgs, cugraph_mfgs):
            # Can not verify edge ids as they are not
            # preserved in cugraph_dgl.nn.SparseGraph
            assert dgl_block.num_src_nodes() == cugraph_dgl_graph.num_src_nodes()
            assert dgl_block.num_dst_nodes() == cugraph_dgl_graph.num_dst_nodes()
            dgl_offsets, dgl_indices, _ = dgl_block.adj_tensors("csc")
            cugraph_offsets, cugraph_indices, _ = cugraph_dgl_graph.csc()
            assert torch.equal(dgl_offsets.to("cpu"), cugraph_offsets.to("cpu"))
            assert torch.equal(dgl_indices.to("cpu"), cugraph_indices.to("cpu"))
