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


from typing import Sequence, Dict, Tuple

from cugraph_pyg.data import CuGraphStore

from cugraph.utilities.utils import import_optional
import cudf

dask_cudf = import_optional("dask_cudf")
torch_geometric = import_optional("torch_geometric")

torch = import_optional("torch")
HeteroSamplerOutput = torch_geometric.sampler.base.HeteroSamplerOutput


def _get_unique_nodes(
    sampling_results: cudf.DataFrame,
    graph_store: CuGraphStore,
    node_type: str,
    node_position: str,
) -> int:
    """
    Counts the number of unique nodes of a given node type.

    Parameters
    ----------
    sampling_results: cudf.DataFrame
        The dataframe containing sampling results or filtered sampling results
        (i.e. sampling results for hop 2)
    graph_store: CuGraphStore
        The graph store containing the structure of the sampled graph.
    node_type: str
        The node type to count the number of unique nodes of.
    node_position: str ('src' or 'dst')
        Whether to examine source or destination nodes.

    Returns
    -------
    cudf.Series
        The unique nodes of the given node type.
    """
    if node_position == "src":
        edge_index = "majors"
        edge_sel = 0
    elif node_position == "dst":
        edge_index = "minors"
        edge_sel = -1
    else:
        raise ValueError(f"Illegal value {node_position} for node_position")

    etypes = [
        graph_store.canonical_edge_type_to_numeric(et)
        for et in graph_store.edge_types
        if et[edge_sel] == node_type
    ]
    if len(etypes) > 0:
        f = sampling_results.edge_type == etypes[0]
        for et in etypes[1:]:
            f |= sampling_results.edge_type == et

        sampling_results_node = sampling_results[f]
    else:
        return cudf.Series([], dtype="int64")

    return sampling_results_node[edge_index]


def _sampler_output_from_sampling_results_homogeneous_coo(
    sampling_results: cudf.DataFrame,
    renumber_map: torch.Tensor,
    graph_store: CuGraphStore,
    data_index: Dict[Tuple[int, int], Dict[str, int]],
    batch_id: int,
    metadata: Sequence = None,
) -> HeteroSamplerOutput:
    """
    Parameters
    ----------
    sampling_results: cudf.DataFrame
        The dataframe containing sampling results.
    renumber_map: torch.Tensor
        The tensor containing the renumber map, or None if there
        is no renumber map.
    graph_store: CuGraphStore
        The graph store containing the structure of the sampled graph.
    data_index: Dict[Tuple[int, int], Dict[str, int]]
        Dictionary where keys are the batch id and hop id,
        and values are dictionaries containing the max src
        and max dst node ids for the batch and hop.
    batch_id: int
        The current batch id, whose samples are being retrieved
        from the sampling results and data index.
    metadata: Tensor
        The metadata for the sampled batch.

    Returns
    -------
    HeteroSamplerOutput
    """

    if len(graph_store.edge_types) > 1 or len(graph_store.node_types) > 1:
        raise ValueError("Graph is heterogeneous")

    hops = torch.arange(
        sampling_results.hop_id.iloc[len(sampling_results) - 1] + 1, device="cuda"
    )
    hops = torch.searchsorted(
        torch.as_tensor(sampling_results.hop_id, device="cuda"), hops
    )

    node_type = graph_store.node_types[0]
    edge_type = graph_store.edge_types[0]

    num_nodes_per_hop_dict = {node_type: torch.zeros(len(hops) + 1, dtype=torch.int64)}
    num_edges_per_hop_dict = {edge_type: torch.zeros(len(hops), dtype=torch.int64)}

    if renumber_map is None:
        raise ValueError("Renumbered input is expected for homogeneous graphs")

    noi_index = {node_type: torch.as_tensor(renumber_map, device="cuda")}

    row_dict = {
        edge_type: torch.as_tensor(sampling_results.majors, device="cuda"),
    }

    col_dict = {
        edge_type: torch.as_tensor(sampling_results.minors, device="cuda"),
    }

    num_nodes_per_hop_dict[node_type][0] = data_index[batch_id, 0]["src_max"] + 1
    for hop in range(len(hops)):
        hop_ix_start = hops[hop]
        hop_ix_end = hops[hop + 1] if hop < len(hops) - 1 else len(sampling_results)

        if num_nodes_per_hop_dict[node_type][hop] > 0:
            max_id_hop = data_index[batch_id, hop]["dst_max"]
            max_id_prev_hop = (
                data_index[batch_id, hop - 1]["dst_max"]
                if hop > 0
                else data_index[batch_id, 0]["src_max"]
            )

            if max_id_hop > max_id_prev_hop:
                num_nodes_per_hop_dict[node_type][hop + 1] = (
                    max_id_hop - max_id_prev_hop
                )
            else:
                num_nodes_per_hop_dict[node_type][hop + 1] = 0
        # will default to 0 if the previous hop was 0, since this is a PyG requirement

        num_edges_per_hop_dict[edge_type][hop] = hop_ix_end - hop_ix_start

    if HeteroSamplerOutput is None:
        raise ImportError("Error importing from pyg")

    return HeteroSamplerOutput(
        node=noi_index,
        row=row_dict,
        col=col_dict,
        edge=None,
        num_sampled_nodes=num_nodes_per_hop_dict,
        num_sampled_edges=num_edges_per_hop_dict,
        metadata=metadata,
    )


def _sampler_output_from_sampling_results_homogeneous_csr(
    major_offsets: torch.Tensor,
    minors: torch.Tensor,
    renumber_map: torch.Tensor,
    graph_store: CuGraphStore,
    label_hop_offsets: torch.Tensor,
    batch_id: int,
    metadata: Sequence = None,
) -> HeteroSamplerOutput:
    """
    Parameters
    ----------
    major_offsets: torch.Tensor
        The major offsets for the CSC/CSR matrix ("row pointer")
    minors: torch.Tensor
        The minors for the CSC/CSR matrix ("col index")
    renumber_map: torch.Tensor
        The tensor containing the renumber map.
        Required.
    graph_store: CuGraphStore
        The graph store containing the structure of the sampled graph.
    label_hop_offsets: torch.Tensor
        The tensor containing the label-hop offsets.
    batch_id: int
        The current batch id, whose samples are being retrieved
        from the sampling results and data index.
    metadata: Tensor
        The metadata for the sampled batch.

    Returns
    -------
    HeteroSamplerOutput
    """

    if len(graph_store.edge_types) > 1 or len(graph_store.node_types) > 1:
        raise ValueError("Graph is heterogeneous")

    if renumber_map is None:
        raise ValueError("Renumbered input is expected for homogeneous graphs")
    node_type = graph_store.node_types[0]
    edge_type = graph_store.edge_types[0]

    major_offsets = major_offsets.clone() - major_offsets[0]
    label_hop_offsets = label_hop_offsets.clone() - label_hop_offsets[0]

    num_edges_per_hop_dict = {edge_type: major_offsets[label_hop_offsets].diff().cpu()}

    label_hop_offsets = label_hop_offsets.cpu()
    num_nodes_per_hop_dict = {
        node_type: torch.concat(
            [
                label_hop_offsets.diff(),
                (renumber_map.shape[0] - label_hop_offsets[-1]).reshape((1,)),
            ]
        ).cpu()
    }

    noi_index = {node_type: torch.as_tensor(renumber_map, device="cuda")}

    col_dict = {
        edge_type: major_offsets,
    }

    row_dict = {
        edge_type: minors,
    }

    if HeteroSamplerOutput is None:
        raise ImportError("Error importing from pyg")

    return HeteroSamplerOutput(
        node=noi_index,
        row=row_dict,
        col=col_dict,
        edge=None,
        num_sampled_nodes=num_nodes_per_hop_dict,
        num_sampled_edges=num_edges_per_hop_dict,
        metadata=metadata,
    )


def _sampler_output_from_sampling_results_heterogeneous(
    sampling_results: cudf.DataFrame,
    renumber_map: cudf.Series,
    graph_store: CuGraphStore,
    metadata: Sequence = None,
) -> HeteroSamplerOutput:
    """
    Parameters
    ----------
    sampling_results: cudf.DataFrame
        The dataframe containing sampling results.
    renumber_map: cudf.Series
        The series containing the renumber map, or None if there
        is no renumber map.
    graph_store: CuGraphStore
        The graph store containing the structure of the sampled graph.
    metadata: Tensor
        The metadata for the sampled batch.

    Returns
    -------
    HeteroSamplerOutput
    """

    hops = torch.arange(sampling_results.hop_id.max() + 1, device="cuda")
    hops = torch.searchsorted(
        torch.as_tensor(sampling_results.hop_id, device="cuda"), hops
    )

    num_nodes_per_hop_dict = {}
    num_edges_per_hop_dict = {}

    # Fill out hop 0 in num_nodes_per_hop_dict, which is based on src instead of dst
    sampling_results_hop_0 = sampling_results.iloc[
        0 : (hops[1] if len(hops) > 1 else len(sampling_results))
    ]

    for node_type in graph_store.node_types:
        num_unique_nodes = _get_unique_nodes(
            sampling_results_hop_0, graph_store, node_type, "src"
        ).nunique()

        if num_unique_nodes > 0:
            num_nodes_per_hop_dict[node_type] = torch.zeros(
                len(hops) + 1, dtype=torch.int64
            )
            num_nodes_per_hop_dict[node_type][0] = num_unique_nodes

    if renumber_map is not None:
        raise ValueError(
            "Precomputing the renumber map is currently "
            "unsupported for heterogeneous graphs."
        )

    # Calculate nodes of interest based on unique nodes in order of appearance
    # Use hop 0 sources since those are the only ones not included in destinations
    # Use torch.concat based on benchmark performance (vs. cudf.concat)

    if sampling_results_hop_0 is None:
        sampling_results_hop_0 = sampling_results.iloc[
            0 : (hops[1] if len(hops) > 1 else len(sampling_results))
        ]

    nodes_of_interest = (
        cudf.Series(
            torch.concat(
                [
                    torch.as_tensor(sampling_results_hop_0.majors, device="cuda"),
                    torch.as_tensor(sampling_results.minors, device="cuda"),
                ]
            ),
            name="nodes_of_interest",
        )
        .drop_duplicates()
        .sort_index()
    )

    # Get the grouped node index (for creating the renumbered grouped edge index)
    noi_index = graph_store._get_vertex_groups_from_sample(
        torch.as_tensor(nodes_of_interest, device="cuda")
    )
    del nodes_of_interest

    # Get the new edge index (by type as expected for HeteroData)
    # FIXME handle edge ids/types after the C++ updates
    row_dict, col_dict = graph_store._get_renumbered_edge_groups_from_sample(
        sampling_results, noi_index
    )

    for hop in range(len(hops)):
        hop_ix_start = hops[hop]
        hop_ix_end = hops[hop + 1] if hop < len(hops) - 1 else len(sampling_results)
        sampling_results_to_hop = sampling_results.iloc[0:hop_ix_end]

        for node_type in graph_store.node_types:
            unique_nodes_hop = _get_unique_nodes(
                sampling_results_to_hop, graph_store, node_type, "dst"
            )

            unique_nodes_0 = _get_unique_nodes(
                sampling_results_hop_0, graph_store, node_type, "src"
            )

            num_unique_nodes = cudf.concat([unique_nodes_0, unique_nodes_hop]).nunique()

            if num_unique_nodes > 0:
                if node_type not in num_nodes_per_hop_dict:
                    num_nodes_per_hop_dict[node_type] = torch.zeros(
                        len(hops) + 1, dtype=torch.int64
                    )
                num_nodes_per_hop_dict[node_type][hop + 1] = num_unique_nodes - int(
                    num_nodes_per_hop_dict[node_type][: hop + 1].sum(0)
                )

        numeric_etypes, counts = torch.unique(
            torch.as_tensor(
                sampling_results.iloc[hop_ix_start:hop_ix_end].edge_type,
                device="cuda",
            ),
            return_counts=True,
        )
        numeric_etypes = list(numeric_etypes)
        counts = list(counts)
        for num_etype, count in zip(numeric_etypes, counts):
            can_etype = graph_store.numeric_edge_type_to_canonical(num_etype)
            if can_etype not in num_edges_per_hop_dict:
                num_edges_per_hop_dict[can_etype] = torch.zeros(
                    len(hops), dtype=torch.int64
                )
            num_edges_per_hop_dict[can_etype][hop] = count

    if HeteroSamplerOutput is None:
        raise ImportError("Error importing from pyg")

    return HeteroSamplerOutput(
        node=noi_index,
        row=row_dict,
        col=col_dict,
        edge=None,
        num_sampled_nodes=num_nodes_per_hop_dict,
        num_sampled_edges=num_edges_per_hop_dict,
        metadata=metadata,
    )

