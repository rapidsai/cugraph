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


def _get_tensor_d_from_sampled_df(df):
    """
    Converts a sampled cuDF DataFrame into a list of tensors.

    Args:
        df (cudf.DataFrame): The sampled cuDF DataFrame containing columns
    Returns:
        dict: A dictionary of tensors, keyed by batch_id and hop_id.
    """
    batch_id_tensor = cast_to_tensor(df["batch_id"])
    batch_id_min = batch_id_tensor.min()
    batch_id_max = batch_id_tensor.max()
    batch_indices = torch.arange(
        start=batch_id_min + 1,
        end=batch_id_max + 1,
        device=batch_id_tensor.device,
    )
    batch_indices = torch.searchsorted(batch_id_tensor, batch_indices)
    split_d = {i: {} for i in range(batch_id_min, batch_id_max + 1)}

    for column in df.columns:
        if column != "batch_id":
            t = cast_to_tensor(df[column])
            split_t = torch.tensor_split(t, batch_indices.cpu())
            for bid, batch_t in zip(split_d.keys(), split_t):
                split_d[bid][column] = batch_t
    del df
    result_tensor_d = {}
    for batch_id, batch_d in split_d.items():
        hop_id_tensor = batch_d["hop_id"]
        hop_id_min = hop_id_tensor.min()
        hop_id_max = hop_id_tensor.max()

        hop_indices = torch.arange(
            start=hop_id_min + 1,
            end=hop_id_max + 1,
            device=hop_id_tensor.device,
        )
        hop_indices = torch.searchsorted(hop_id_tensor, hop_indices)
        hop_split_d = {i: {} for i in range(hop_id_min, hop_id_max + 1)}
        for column, t in batch_d.items():
            if column != "hop_id":
                split_t = torch.tensor_split(t, hop_indices.cpu())
                for hid, ht in zip(hop_split_d.keys(), split_t):
                    hop_split_d[hid][column] = ht

        result_tensor_d[batch_id] = hop_split_d
    return result_tensor_d


def create_homogeneous_sampled_graphs_from_dataframe(
    sampled_df: cudf.DataFrame,
    total_number_of_nodes: int,
    edge_dir: str = "in",
):
    """
    This helper function creates DGL MFGS  for
    homogeneous graphs from cugraph sampled dataframe
    """
    result_tensor_d = _get_tensor_d_from_sampled_df(sampled_df)
    del sampled_df
    result_mfgs = [
        _create_homogeneous_sampled_graphs_from_tensors_perhop(
            tensors_batch_d, total_number_of_nodes, edge_dir
        )
        for tensors_batch_d in result_tensor_d.values()
    ]
    del result_tensor_d
    return result_mfgs


def _create_homogeneous_sampled_graphs_from_tensors_perhop(
    tensors_batch_d, total_number_of_nodes, edge_dir
):
    if edge_dir not in ["in", "out"]:
        raise ValueError(f"Invalid edge_dir {edge_dir} provided")
    if edge_dir == "out":
        raise ValueError("Outwards edges not supported yet")
    graph_per_hop_ls = []
    output_nodes = None
    seed_nodes = None
    # st = time.time()
    for hop_id, tensor_d in tensors_batch_d.items():
        block = _create_homogeneous_dgl_block_from_tensor_d(tensor_d)
        seed_nodes = tensor_d["sources"]
        if output_nodes is None:
            output_nodes = tensor_d["destinations"]
        graph_per_hop_ls.append(block)
    # et = time.time()
    # print("Time to create blocks", et - st)

    # default DGL behavior
    if edge_dir == "in":
        graph_per_hop_ls.reverse()
    return seed_nodes.unique(), output_nodes.unique(), graph_per_hop_ls


def _create_homogeneous_dgl_block_from_tensor_d(
    tensor_d,
):
    rs = tensor_d["renumbered_sources"]
    rd = tensor_d["renumbered_destinations"]
    block = dgl.create_block((rs, rd))
    block.edata[dgl.EID] = tensor_d["edge_id"]
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
    # Optimize below function
    # based on _get_tensor_ls_from_sampled_df
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
