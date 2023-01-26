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


def create_cugraph_dgl_homogenous_mfgs(g, seed_nodes, fanout):
    df_ls = []
    for hop_id, fanout in enumerate(reversed(fanout)):
        frontier = g.sample_neighbors(seed_nodes, fanout)
        # Set include_dst_in_src to match cugraph behavior
        block = dgl.to_block(frontier, seed_nodes, include_dst_in_src=False)
        block.edata[dgl.EID] = frontier.edata[dgl.EID]
        seed_nodes = block.srcdata[dgl.NID]
        block = block.to("cpu")
        src, dst, eid = block.edges("all")
        src = block.srcdata[dgl.NID][src]
        dst = block.dstdata[dgl.NID][dst]
        eid = block.edata[dgl.EID][eid]
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
    return create_homogeneous_sampled_graphs_from_dataframe(df, g.num_nodes())[0]


@pytest.mark.parametrize("seed_node", [3, 4, 5])
def test_homogeneous_sampled_graphs_from_dataframe(seed_node):
    g = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 5]))
    fanout = [1, 1, 1]
    seed_node = torch.as_tensor([seed_node])

    dgl_seed_nodes, dgl_output_nodes, dgl_mfgs = create_cugraph_dgl_homogenous_mfgs(
        g, seed_node, fanout
    )
    (
        cugraph_seed_nodes,
        cugraph_output_nodes,
        cugraph_mfgs,
    ) = create_cugraph_dgl_homogenous_mfgs(g, seed_node, fanout)

    np.testing.assert_equal(
        cugraph_seed_nodes.cpu().numpy().copy().sort(),
        dgl_seed_nodes.cpu().numpy().copy().sort(),
    )

    np.testing.assert_equal(
        dgl_output_nodes.cpu().numpy().copy().sort(),
        cugraph_output_nodes.cpu().numpy().copy().sort(),
    )

    for dgl_block, cugraph_dgl_block in zip(dgl_mfgs, cugraph_mfgs):
        dgl_df = get_edge_df_from_homogenous_block(dgl_block)
        cugraph_dgl_df = get_edge_df_from_homogenous_block(cugraph_dgl_block)
        pd.testing.assert_frame_equal(dgl_df, cugraph_dgl_df)
