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
from __future__ import annotations
from typing import Tuple, Dict, Optional
from collections import defaultdict
import cudf
from cugraph.utilities.utils import import_optional

dgl = import_optional("dgl")
torch = import_optional("torch")


def cast_to_tensor(ser: cudf.Series):
    if len(ser) == 0:
        # Empty series can not be converted to pytorch cuda tensor
        t = torch.from_numpy(ser.values.get())
        return t.to("cuda")

    return torch.as_tensor(ser.values, device="cuda")


def create_homogeneous_sampled_graphs_from_dataframe(
    sampled_df: cudf.DataFrame,
    total_number_of_nodes: int,
    edge_dir: str = "in",
):
    """
    This helper function creates DGL MFGS  for
    homogeneous graphs from cugraph sampled dataframe
    """

    sampled_df["batch_id"] = sampled_df["batch_id"] - sampled_df["batch_id"].min()
    result_df_ls = sampled_df[
        ["sources", "destinations", "edge_id", "hop_id"]
    ].scatter_by_map(sampled_df["batch_id"], keep_index=False)
    del sampled_df
    result_df_ls = [
        batch_df[["sources", "destinations", "edge_id"]].scatter_by_map(
            batch_df["hop_id"], keep_index=False
        )
        for batch_df in result_df_ls
    ]

    result_tensor_ls = [
        [
            (
                cast_to_tensor(h_df["sources"]),
                cast_to_tensor(h_df["destinations"]),
                cast_to_tensor(h_df["edge_id"]),
            )
            for h_df in per_batch_ls
        ]
        for per_batch_ls in result_df_ls
    ]
    del result_df_ls
    result_mfgs = [
        _create_homogeneous_sampled_graphs_from_tensors_perhop(
            tensors_perhop_ls, total_number_of_nodes, edge_dir
        )
        for tensors_perhop_ls in result_tensor_ls
    ]
    del result_tensor_ls
    return result_mfgs


def _create_homogeneous_sampled_graphs_from_tensors_perhop(
    tensors_perhop_ls, total_number_of_nodes, edge_dir
):
    if edge_dir not in ["in", "out"]:
        raise ValueError(f"Invalid edge_dir {edge_dir} provided")
    if edge_dir == "out":
        raise ValueError("Outwards edges not supported yet")
    graph_per_hop_ls = []
    output_nodes = None
    seed_nodes = None
    for src_ids, dst_ids, edge_ids in tensors_perhop_ls:
        # print("Creating block", flush=True)
        block = create_homogeneous_dgl_block_from_tensors_ls(
            src_ids=src_ids,
            dst_ids=dst_ids,
            edge_ids=edge_ids,
            seed_nodes=seed_nodes,
            total_number_of_nodes=total_number_of_nodes,
        )
        seed_nodes = block.srcdata[dgl.NID]
        if output_nodes is None:
            output_nodes = block.dstdata[dgl.NID]
        graph_per_hop_ls.append(block)

    # default DGL behavior
    if edge_dir == "in":
        graph_per_hop_ls.reverse()
    return seed_nodes, output_nodes, graph_per_hop_ls


def create_homogeneous_dgl_block_from_tensors_ls(
    src_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    seed_nodes: Optional[torch.Tensor],
    total_number_of_nodes: int,
):
    sampled_graph = dgl.graph(
        (src_ids, dst_ids),
        num_nodes=total_number_of_nodes,
    )
    sampled_graph.edata[dgl.EID] = edge_ids
    # TODO: Check if unique is needed
    if seed_nodes is None:
        seed_nodes = dst_ids.unique()

    block = dgl.to_block(
        sampled_graph,
        dst_nodes=seed_nodes,
        src_nodes=src_ids.unique(),
        include_dst_in_src=True,
    )
    block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    return block


def create_heterogeneous_sampled_graphs_from_dataframe(
    sampled_df: cudf.DataFrame,
    num_nodes_dict: Dict[str, int],
    etype_id_dict: Dict[int, Tuple[str, str, str]],
    etype_offset_dict: Dict[Tuple[str, str, str], int],
    ntype_offset_dict: Dict[str, int],
    edge_dir: str = "in",
):
    """
    This helper function creates DGL MFGS from cugraph sampled dataframe
    """
    sampled_df["batch_id"] = sampled_df["batch_id"] - sampled_df["batch_id"].min()
    result_df_ls = sampled_df[
        ["sources", "destinations", "edge_id", "hop_id", "edge_type"]
    ].scatter_by_map(sampled_df["batch_id"], keep_index=False)
    del sampled_df

    result_df_ls = [
        batch_df[["sources", "destinations", "edge_id", "edge_type"]].scatter_by_map(
            batch_df["hop_id"], keep_index=False
        )
        for batch_df in result_df_ls
    ]

    result_tensor_ls = [
        [
            _get_edges_dict_from_perhop_df(
                h_df, etype_id_dict, etype_offset_dict, ntype_offset_dict
            )
            for h_df in per_batch_ls
        ]
        for per_batch_ls in result_df_ls
    ]
    del result_df_ls

    result_mfgs = [
        _create_heterogenous_sampled_graphs_from_tensors_perhop(
            tensors_perhop_ls, num_nodes_dict, edge_dir
        )
        for tensors_perhop_ls in result_tensor_ls
    ]
    return result_mfgs


def _get_edges_dict_from_perhop_df(
    df, etype_id_dict, etype_offset_dict, ntype_offset_dict
):
    edges_per_type_ls = df[["sources", "destinations", "edge_id"]].scatter_by_map(
        df["edge_type"], map_size=len(etype_id_dict), keep_index=False
    )
    del df
    per_type_df_d = {etype_id_dict[i]: df for i, df in enumerate(edges_per_type_ls)}
    del edges_per_type_ls
    # reverse src,dst here
    per_type_tensor_d = {
        etype: (
            cast_to_tensor(etype_df["sources"]) - ntype_offset_dict[etype[0]],
            cast_to_tensor(etype_df["destinations"]) - ntype_offset_dict[etype[2]],
            cast_to_tensor(etype_df["edge_id"]) - etype_offset_dict[etype],
        )
        for etype, etype_df in per_type_df_d.items()
    }
    return per_type_tensor_d


def _create_heterogenous_sampled_graphs_from_tensors_perhop(
    tensors_perhop_ls, num_nodes_dict, edge_dir
):
    if edge_dir not in ["in", "out"]:
        raise ValueError(f"Invalid edge_dir {edge_dir} provided")
    if edge_dir == "out":
        raise ValueError("Outwards edges not supported yet")
    graph_per_hop_ls = []
    output_nodes = None

    seed_nodes = None
    for hop_edges_dict in tensors_perhop_ls:
        block = create_heterogenous_dgl_block_from_tensors_dict(
            hop_edges_dict, num_nodes_dict, seed_nodes
        )
        seed_nodes = block.srcdata[dgl.NID]
        if output_nodes is None:
            output_nodes = block.dstdata[dgl.NID]
        graph_per_hop_ls.append(block)

    # default DGL behavior
    if edge_dir == "in":
        graph_per_hop_ls.reverse()
    return seed_nodes, output_nodes, graph_per_hop_ls


def create_heterogenous_dgl_block_from_tensors_dict(
    edges_dict: Dict[Tuple(str, str, str), (torch.Tensor, torch.Tensor, torch.Tensor)],
    num_nodes_dict: Dict[str, torch.Tensor],
    seed_nodes: Optional[Dict[str, torch.Tensor]],
):
    data_dict = {k: (s, d) for k, (s, d, _) in edges_dict.items()}
    edge_ids_dict = {k: eid for k, (_, _, eid) in edges_dict.items()}

    sampled_graph = dgl.heterograph(
        data_dict=data_dict,
        num_nodes_dict=num_nodes_dict,
    )
    sampled_graph.edata[dgl.EID] = edge_ids_dict

    src_d = defaultdict(list)
    dst_d = defaultdict(list)

    for (s, _, d), (src_id, dst_id) in data_dict.items():
        src_d[s].append(src_id)
        dst_d[d].append(dst_id)

    src_d = {k: torch.cat(v).unique() for k, v in src_d.items() if len(v) > 0}
    if seed_nodes is None:
        seed_nodes = {k: torch.cat(v).unique() for k, v in dst_d.items() if len(v) > 0}

    block = dgl.to_block(sampled_graph, dst_nodes=seed_nodes, src_nodes=src_d)
    block.edata[dgl.EID] = sampled_graph.edata[dgl.EID]
    return block
