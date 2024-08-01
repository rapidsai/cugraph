# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import cudf
from cugraph.utilities.utils import import_optional
from cugraph_dgl.nn import SparseGraph

dgl = import_optional("dgl")
torch = import_optional("torch")
cugraph_dgl = import_optional("cugraph_dgl")


def cast_to_tensor(ser: cudf.Series):
    if len(ser) == 0:
        # Empty series can not be converted to pytorch cuda tensor
        t = torch.from_numpy(ser.values.get())
        return t.to("cuda")

    return torch.as_tensor(ser.values, device="cuda")


def _split_tensor(t, split_indices):
    """
    Split a tensor into a list of tensors based on split_indices.
    """
    # TODO: Switch to something below
    # return [t[i:j] for i, j in zip(split_indices[:-1], split_indices[1:])]
    if split_indices.device.type != "cpu":
        split_indices = split_indices.to("cpu")
    return torch.tensor_split(t, split_indices)


def _get_source_destination_range(sampled_df):
    o = sampled_df.groupby(["batch_id", "hop_id"], as_index=True).agg(
        {"sources": "max", "destinations": "max"}
    )
    o.rename(
        columns={"sources": "sources_range", "destinations": "destinations_range"},
        inplace=True,
    )
    d = o.to_dict(orient="index")
    return d


def _create_split_dict(tensor):
    min_value = tensor.min()
    max_value = tensor.max()
    indices = torch.arange(
        start=min_value + 1,
        end=max_value + 1,
        device=tensor.device,
    )
    split_dict = {i: {} for i in range(min_value, max_value + 1)}
    return split_dict, indices


def _get_renumber_map(df):
    map = df["map"]
    df.drop(columns=["map"], inplace=True)

    map_starting_offset = map.iloc[0]
    renumber_map = map[map_starting_offset:].dropna().reset_index(drop=True)
    renumber_map_batch_indices = map[1 : map_starting_offset - 1].reset_index(drop=True)
    renumber_map_batch_indices = renumber_map_batch_indices - map_starting_offset

    map_end_offset = map_starting_offset + len(renumber_map)
    # We only need to drop rows if the length of dataframe is determined by the map
    # that is if map_length > sampled edges length
    if map_end_offset == len(df):
        df.dropna(axis=0, how="all", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df, cast_to_tensor(renumber_map), cast_to_tensor(renumber_map_batch_indices)


def _get_tensor_d_from_sampled_df(df):
    """
    Converts a sampled cuDF DataFrame into a list of tensors.

    Args:
        df (cudf.DataFrame): The sampled cuDF DataFrame containing columns
    Returns:
        dict: A dictionary of tensors, keyed by batch_id and hop_id.
    """
    range_d = _get_source_destination_range(df)
    df, renumber_map, renumber_map_batch_indices = _get_renumber_map(df)
    batch_id_tensor = cast_to_tensor(df["batch_id"])
    split_d, batch_indices = _create_split_dict(batch_id_tensor)
    batch_split_indices = torch.searchsorted(batch_id_tensor, batch_indices).to("cpu")

    for column in df.columns:
        if column != "batch_id":
            t = cast_to_tensor(df[column])
            split_t = _split_tensor(t, batch_split_indices)
            for bid, batch_t in zip(split_d.keys(), split_t):
                split_d[bid][column] = batch_t

    split_t = _split_tensor(renumber_map, renumber_map_batch_indices)
    for bid, batch_t in zip(split_d.keys(), split_t):
        split_d[bid]["map"] = batch_t
    del df
    result_tensor_d = {}
    # Cache hop_split_d, hop_indices
    hop_split_empty_d, hop_indices = None, None
    for batch_id, batch_d in split_d.items():
        hop_id_tensor = batch_d["hop_id"]
        if hop_split_empty_d is None:
            hop_split_empty_d, hop_indices = _create_split_dict(hop_id_tensor)

        hop_split_d = {k: {} for k in hop_split_empty_d.keys()}
        hop_split_indices = torch.searchsorted(hop_id_tensor, hop_indices).to("cpu")
        for column, t in batch_d.items():
            if column not in ["hop_id", "map"]:
                split_t = _split_tensor(t, hop_split_indices)
                for hid, ht in zip(hop_split_d.keys(), split_t):
                    hop_split_d[hid][column] = ht
        for hid in hop_split_d.keys():
            hop_split_d[hid]["sources_range"] = range_d[(batch_id, hid)][
                "sources_range"
            ]
            hop_split_d[hid]["destinations_range"] = range_d[(batch_id, hid)][
                "destinations_range"
            ]

        result_tensor_d[batch_id] = hop_split_d
        result_tensor_d[batch_id]["map"] = batch_d["map"]
    return result_tensor_d


def create_homogeneous_sampled_graphs_from_dataframe(
    sampled_df: cudf.DataFrame,
    edge_dir: str = "in",
    return_type: str = "dgl.Block",
):
    """
    This helper function creates DGL MFGS  for
    homogeneous graphs from cugraph sampled dataframe

    Args:
        sampled_df (cudf.DataFrame): The sampled cuDF DataFrame containing
            columns `sources`, `destinations`, `edge_id`, `batch_id` and
            `hop_id`.
        edge_dir (str): Direction of edges from samples
    Returns:
        list: A list containing three elements:
            - input_nodes: The input nodes for the batch.
            - output_nodes: The output nodes for the batch.
            - graph_per_hop_ls: A list of DGL MFGS for each hop.
    """
    if return_type not in ["dgl.Block", "cugraph_dgl.nn.SparseGraph"]:
        raise ValueError(
            "return_type must be either dgl.Block or cugraph_dgl.nn.SparseGraph"
        )

    result_tensor_d = _get_tensor_d_from_sampled_df(sampled_df)
    del sampled_df
    result_mfgs = [
        _create_homogeneous_sampled_graphs_from_tensors_perhop(
            tensors_batch_d, edge_dir, return_type
        )
        for tensors_batch_d in result_tensor_d.values()
    ]
    del result_tensor_d
    return result_mfgs


def _create_homogeneous_sampled_graphs_from_tensors_perhop(
    tensors_batch_d, edge_dir, return_type
):
    """
    This helper function creates sampled DGL MFGS for
    homogeneous graphs from tensors per hop for a single
    batch
    Args:
        tensors_batch_d (dict): A dictionary of tensors, keyed by hop_id.
        edge_dir (str): Direction of edges from samples
        metagraph (dgl.metagraph): The metagraph for the sampled graph
        return_type (str): The type of graph to return
    Returns:
        tuple: A tuple of three elements:
            - input_nodes: The input nodes for the batch.
            - output_nodes: The output nodes for the batch.
            - graph_per_hop_ls: A list of MFGS for each hop.
    """
    if edge_dir not in ["in", "out"]:
        raise ValueError(f"Invalid edge_dir {edge_dir} provided")
    if edge_dir == "out":
        raise ValueError("Outwards edges not supported yet")
    graph_per_hop_ls = []
    seednodes_range = None
    for hop_id, tensor_per_hop_d in tensors_batch_d.items():
        if hop_id != "map":
            if return_type == "dgl.Block":
                mfg = _create_homogeneous_dgl_block_from_tensor_d(
                    tensor_d=tensor_per_hop_d,
                    renumber_map=tensors_batch_d["map"],
                    seednodes_range=seednodes_range,
                )
            elif return_type == "cugraph_dgl.nn.SparseGraph":
                mfg = _create_homogeneous_cugraph_dgl_nn_sparse_graph(
                    tensor_d=tensor_per_hop_d, seednodes_range=seednodes_range
                )
            else:
                raise ValueError(f"Invalid return_type {return_type} provided")
            seednodes_range = max(
                tensor_per_hop_d["sources_range"],
                tensor_per_hop_d["destinations_range"],
            )
            graph_per_hop_ls.append(mfg)

    # default DGL behavior
    if edge_dir == "in":
        graph_per_hop_ls.reverse()
    if return_type == "dgl.Block":
        input_nodes = graph_per_hop_ls[0].srcdata[dgl.NID]
        output_nodes = graph_per_hop_ls[-1].dstdata[dgl.NID]
    else:
        map = tensors_batch_d["map"]
        input_nodes = map[0 : graph_per_hop_ls[0].num_src_nodes()]
        output_nodes = map[0 : graph_per_hop_ls[-1].num_dst_nodes()]
    return input_nodes, output_nodes, graph_per_hop_ls


def _create_homogeneous_dgl_block_from_tensor_d(
    tensor_d,
    renumber_map,
    seednodes_range=None,
):
    rs = tensor_d["sources"]
    rd = tensor_d["destinations"]
    max_src_nodes = tensor_d["sources_range"]
    max_dst_nodes = tensor_d["destinations_range"]
    if seednodes_range is not None:
        # If we have  vertices without outgoing edges, then
        # sources can be missing from seednodes
        # so we add them
        # to ensure all the blocks are
        # lined up correctly
        max_dst_nodes = max(max_dst_nodes, seednodes_range)

    data_dict = {("_N", "_E", "_N"): (rs, rd)}
    num_src_nodes = {"_N": max_src_nodes + 1}
    num_dst_nodes = {"_N": max_dst_nodes + 1}

    block = dgl.create_block(
        data_dict=data_dict, num_src_nodes=num_src_nodes, num_dst_nodes=num_dst_nodes
    )
    if "edge_id" in tensor_d:
        block.edata[dgl.EID] = tensor_d["edge_id"]
    # Below adds run time overhead
    block.srcdata[dgl.NID] = renumber_map[0 : max_src_nodes + 1]
    block.dstdata[dgl.NID] = renumber_map[0 : max_dst_nodes + 1]
    return block


def _create_homogeneous_cugraph_dgl_nn_sparse_graph(tensor_d, seednodes_range):
    max_src_nodes = tensor_d["sources_range"]
    max_dst_nodes = tensor_d["destinations_range"]
    if seednodes_range is not None:
        max_dst_nodes = max(max_dst_nodes, seednodes_range)
    size = (max_src_nodes + 1, max_dst_nodes + 1)
    sparse_graph = cugraph_dgl.nn.SparseGraph(
        size=size,
        src_ids=tensor_d["sources"],
        dst_ids=tensor_d["destinations"],
        formats=["csc"],
        reduce_memory=True,
    )
    return sparse_graph


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


def _process_sampled_tensors_csc(
    tensors: Dict["torch.Tensor"],
    reverse_hop_id: bool = True,
) -> Tuple[
    Dict[int, Dict[int, Dict[str, "torch.Tensor"]]],
    List["torch.Tensor"],
    List[List[int, int]],
]:
    """
    Convert tensors generated by BulkSampler to a dictionary of tensors, to
    facilitate MFG creation. The sampled graphs in the dataframe use CSC-format.

    Parameters
    ----------
    tensors: Dict[torch.Tensor]
        The output from BulkSampler compressed in CSC format. The dataframe
        should be generated with `compression="CSR"` in BulkSampler,
        since the sampling routine treats seed nodes as sources.

    reverse_hop_id: bool (default=True)
        Reverse hop id.

    Returns
    -------
    tensors_dict: dict
        A nested dictionary keyed by batch id and hop id.
        `tensor_dict[batch_id][hop_id]` holds "minors" and "major_offsets"
        values for CSC MFGs.

    renumber_map_list: list
        List of renumbering maps for looking up global indices of nodes. One
        map for each batch.

    mfg_sizes: list
        List of the number of nodes in each message passing layer. For the
        k-th hop, mfg_sizes[k] and mfg_sizes[k+1] is the number of sources and
        destinations, respectively.
    """

    major_offsets = tensors["major_offsets"]
    minors = tensors["minors"]
    label_hop_offsets = tensors["label_hop_offsets"]
    renumber_map = tensors["map"]
    renumber_map_offsets = tensors["renumber_map_offsets"]

    n_batches = len(renumber_map_offsets) - 1
    n_hops = int((len(label_hop_offsets) - 1) / n_batches)

    # make global offsets local
    # Have to make a clone as pytorch does not allow
    # in-place operations on tensors
    major_offsets -= major_offsets[0].clone()
    label_hop_offsets -= label_hop_offsets[0].clone()
    renumber_map_offsets -= renumber_map_offsets[0].clone()

    # get the sizes of each adjacency matrix (for MFGs)
    mfg_sizes = (label_hop_offsets[1:] - label_hop_offsets[:-1]).reshape(
        (n_batches, n_hops)
    )
    n_nodes = renumber_map_offsets[1:] - renumber_map_offsets[:-1]
    mfg_sizes = torch.hstack((mfg_sizes, n_nodes.reshape(n_batches, -1)))
    if reverse_hop_id:
        mfg_sizes = mfg_sizes.flip(1)

    tensors_dict = {}
    renumber_map_list = []
    # Note: minors and major_offsets from BulkSampler are of type int32
    # and int64 respectively. Since pylibcugraphops binding code doesn't
    # support distinct node and edge index type, we simply casting both
    # to int32 for now.
    minors = minors.int()
    major_offsets = major_offsets.int()
    # Note: We transfer tensors to CPU here to avoid the overhead of
    # transferring them in each iteration of the for loop below.
    major_offsets_cpu = major_offsets.to("cpu").numpy()
    label_hop_offsets_cpu = label_hop_offsets.to("cpu").numpy()

    for batch_id in range(n_batches):
        batch_dict = {}
        for hop_id in range(n_hops):
            hop_dict = {}
            idx = batch_id * n_hops + hop_id  # idx in label_hop_offsets
            major_offsets_start = label_hop_offsets_cpu[idx]
            major_offsets_end = label_hop_offsets_cpu[idx + 1]
            minors_start = major_offsets_cpu[major_offsets_start]
            minors_end = major_offsets_cpu[major_offsets_end]
            hop_dict["minors"] = minors[minors_start:minors_end]
            hop_dict["major_offsets"] = (
                major_offsets[major_offsets_start : major_offsets_end + 1]
                - major_offsets[major_offsets_start]
            )
            if reverse_hop_id:
                batch_dict[n_hops - 1 - hop_id] = hop_dict
            else:
                batch_dict[hop_id] = hop_dict

        tensors_dict[batch_id] = batch_dict

        renumber_map_list.append(
            renumber_map[
                renumber_map_offsets[batch_id] : renumber_map_offsets[batch_id + 1]
            ],
        )

    return tensors_dict, renumber_map_list, mfg_sizes.tolist()


def _process_sampled_df_csc(
    df: cudf.DataFrame,
    reverse_hop_id: bool = True,
):
    """
    Convert a dataframe generated by BulkSampler to a dictionary of tensors, to
    facilitate MFG creation. The sampled graphs in the dataframe use CSC-format.

    Parameters
    ----------
    df: cudf.DataFrame
        The output from BulkSampler compressed in CSC format. The dataframe
        should be generated with `compression="CSR"` in BulkSampler,
        since the sampling routine treats seed nodes as sources.

    reverse_hop_id: bool (default=True)
        Reverse hop id.

    Returns
    -------
    tensors_dict: dict
        A nested dictionary keyed by batch id and hop id.
        `tensor_dict[batch_id][hop_id]` holds "minors" and "major_offsets"
        values for CSC MFGs.

    renumber_map_list: list
        List of renumbering maps for looking up global indices of nodes. One
        map for each batch.

    mfg_sizes: list
        List of the number of nodes in each message passing layer. For the
        k-th hop, mfg_sizes[k] and mfg_sizes[k+1] is the number of sources and
        destinations, respectively.
    """

    return _process_sampled_tensors_csc(
        {
            "major_offsets": cast_to_tensor(df.major_offsets.dropna()),
            "label_hop_offsets": cast_to_tensor(df.label_hop_offsets.dropna()),
            "renumber_map_offsets": cast_to_tensor(df.renumber_map_offsets.dropna()),
            "map": cast_to_tensor(df["map"].dropna()),
            "minors": cast_to_tensor(df.minors.dropna()),
        },
        reverse_hop_id=reverse_hop_id,
    )


def _create_homogeneous_blocks_from_csc(
    tensors_dict: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    renumber_map_list: List[torch.Tensor],
    mfg_sizes: List[int, int],
):
    """Create mini-batches of MFGs in the dgl.Block format.
    The input arguments are the outputs of
    the function `_process_sampled_df_csc`.

    Returns
    -------
    output: list
        A list of mini-batches. Each mini-batch is a list that consists of
        `input_nodes` tensor, `output_nodes` tensor and a list of MFGs.
    """
    n_batches, n_hops = len(mfg_sizes), len(mfg_sizes[0]) - 1
    output = []
    for b_id in range(n_batches):
        output_batch = []
        output_batch.append(renumber_map_list[b_id])
        output_batch.append(renumber_map_list[b_id][: mfg_sizes[b_id][-1]])

        mfgs = [
            SparseGraph(
                size=(mfg_sizes[b_id][h_id], mfg_sizes[b_id][h_id + 1]),
                src_ids=tensors_dict[b_id][h_id]["minors"],
                cdst_ids=tensors_dict[b_id][h_id]["major_offsets"],
                formats=["csc", "coo"],
                reduce_memory=True,
            )
            for h_id in range(n_hops)
        ]

        blocks = []
        seednodes_range = None
        for mfg in reversed(mfgs):
            block_mfg = _create_homogeneous_dgl_block_from_tensor_d(
                {
                    "sources": mfg.src_ids(),
                    "destinations": mfg.dst_ids(),
                    "sources_range": mfg._num_src_nodes - 1,
                    "destinations_range": mfg._num_dst_nodes - 1,
                },
                renumber_map=renumber_map_list[b_id],
                seednodes_range=seednodes_range,
            )

            seednodes_range = max(
                mfg._num_src_nodes - 1,
                mfg._num_dst_nodes - 1,
            )
            blocks.append(block_mfg)
        del mfgs

        blocks.reverse()

        output_batch.append(blocks)

        output.append(output_batch)
    return output


def _create_homogeneous_sparse_graphs_from_csc(
    tensors_dict: Dict[int, Dict[int, Dict[str, torch.Tensor]]],
    renumber_map_list: List[torch.Tensor],
    mfg_sizes: List[int, int],
) -> List[List[torch.Tensor, torch.Tensor, List[SparseGraph]]]:
    """Create mini-batches of MFGs. The input arguments are the outputs of
    the function `_process_sampled_df_csc`.

    Returns
    -------
    output: list
        A list of mini-batches. Each mini-batch is a list that consists of
        `input_nodes` tensor, `output_nodes` tensor and a list of MFGs.
    """
    n_batches, n_hops = len(mfg_sizes), len(mfg_sizes[0]) - 1
    output = []
    for b_id in range(n_batches):
        output_batch = []
        output_batch.append(renumber_map_list[b_id])
        output_batch.append(renumber_map_list[b_id][: mfg_sizes[b_id][-1]])
        mfgs = [
            SparseGraph(
                size=(mfg_sizes[b_id][h_id], mfg_sizes[b_id][h_id + 1]),
                src_ids=tensors_dict[b_id][h_id]["minors"],
                cdst_ids=tensors_dict[b_id][h_id]["major_offsets"],
                formats=["csc"],
                reduce_memory=True,
            )
            for h_id in range(n_hops)
        ]

        output_batch.append(mfgs)

        output.append(output_batch)

    return output


def create_homogeneous_sampled_graphs_from_dataframe_csc(
    sampled_df: cudf.DataFrame, output_format: str = "cugraph_dgl.nn.SparseGraph"
):
    """Public API to create mini-batches of MFGs using a dataframe output by
    BulkSampler, where the sampled graph is compressed in CSC format."""
    if output_format == "cugraph_dgl.nn.SparseGraph":
        return _create_homogeneous_sparse_graphs_from_csc(
            *(_process_sampled_df_csc(sampled_df)),
        )
    elif output_format == "dgl.Block":
        return _create_homogeneous_blocks_from_csc(
            *(_process_sampled_df_csc(sampled_df)),
        )
    else:
        raise ValueError(f"Invalid output format {output_format}")


def create_homogeneous_sampled_graphs_from_tensors_csc(
    tensors: Dict["torch.Tensor"], output_format: str = "cugraph_dgl.nn.SparseGraph"
):
    """Public API to create mini-batches of MFGs using a dataframe output by
    BulkSampler, where the sampled graph is compressed in CSC format."""
    if output_format == "cugraph_dgl.nn.SparseGraph":
        return _create_homogeneous_sparse_graphs_from_csc(
            *(_process_sampled_tensors_csc(tensors)),
        )
    elif output_format == "dgl.Block":
        return _create_homogeneous_blocks_from_csc(
            *(_process_sampled_tensors_csc(tensors)),
        )
    else:
        raise ValueError(f"Invalid output format {output_format}")
