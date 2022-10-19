# Copyright (c) 2022, NVIDIA CORPORATION.
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


# Utils for sampling on graphstore like objects
import cugraph
import cudf
import cupy as cp
import dask_cudf
from cugraph.experimental import PropertyGraph

src_n = PropertyGraph.src_col_name
dst_n = PropertyGraph.dst_col_name
type_n = PropertyGraph.type_col_name
eid_n = PropertyGraph.edge_id_col_name
vid_n = PropertyGraph.vertex_col_name


def get_subgraph_and_src_range_from_edgelist(edge_list, is_mg, reverse_edges=False):
    if reverse_edges:
        edge_list = edge_list.rename(columns={src_n: dst_n, dst_n: src_n})

    subgraph = cugraph.MultiGraph(directed=True)
    if is_mg:
        # FIXME: Can not switch to renumber = False
        # For MNMG Algos
        # Remove when https://github.com/rapidsai/cugraph/issues/2437
        # lands
        create_subgraph_f = subgraph.from_dask_cudf_edgelist
        renumber = True
        edge_list = edge_list.persist()
        src_range = edge_list[src_n].min().compute(), edge_list[src_n].max().compute()

    else:
        # Note: We have to keep renumber = False
        # to handle cases when the seed_nodes is not present in subgraph
        create_subgraph_f = subgraph.from_cudf_edgelist
        renumber = False
        src_range = edge_list[src_n].min(), edge_list[src_n].max()

    create_subgraph_f(
        edge_list,
        source=src_n,
        destination=dst_n,
        edge_attr=eid_n,
        renumber=renumber,
        # FIXME: renumber=False is not supported for MNMG algos
        legacy_renum_only=True,
    )

    return subgraph, src_range


def sample_multiple_sgs(
    sgs,
    sample_f,
    start_list_d,
    start_list_dtype,
    edge_dir,
    fanout,
    with_replacement,
):
    start_list_types = list(start_list_d.keys())
    output_dfs = []
    for can_etype, (sg, start_list_range) in sgs.items():
        can_etype = _convert_can_etype_s_to_tup(can_etype)
        if _edge_types_contains_canonical_etype(can_etype, start_list_types, edge_dir):
            if edge_dir == "in":
                subset_type = can_etype[2]
            else:
                subset_type = can_etype[0]
            output = sample_single_sg(
                sg,
                sample_f,
                start_list_d[subset_type],
                start_list_dtype,
                start_list_range,
                fanout,
                with_replacement,
            )
            output_dfs.append(output)
    if len(output_dfs) == 0:
        empty_df = cudf.DataFrame({"sources": [], "destinations": [], "indices": []})
        return empty_df.astype(cp.int32)

    if isinstance(output_dfs[0], dask_cudf.DataFrame):
        return dask_cudf.concat(output_dfs, ignore_index=True)
    else:
        return cudf.concat(output_dfs, ignore_index=True)


def sample_single_sg(
    sg,
    sample_f,
    start_list,
    start_list_dtype,
    start_list_range,
    fanout,
    with_replacement,
):
    if isinstance(start_list, dict):
        start_list = cudf.concat(list(start_list.values()))

    # Uniform sampling fails when the dtype
    # of the seed dtype is not same as the node dtype
    start_list = start_list.astype(start_list_dtype)

    # Filter start list by ranges
    # to enure the seed is with in index values
    # see below:
    # https://github.com/rapidsai/cugraph/blob/branch-22.12/cpp/src/prims/per_v_random_select_transform_outgoing_e.cuh
    start_list = start_list[
        (start_list >= start_list_range[0]) & (start_list <= start_list_range[1])
    ]
    sampled_df = sample_f(
        sg,
        start_list=start_list,
        fanout_vals=[fanout],
        with_replacement=with_replacement,
    )
    return sampled_df


def _edge_types_contains_canonical_etype(can_etype, edge_types, edge_dir):
    src_type, _, dst_type = can_etype
    if edge_dir == "in":
        return dst_type in edge_types
    else:
        return src_type in edge_types


def _convert_can_etype_s_to_tup(canonical_etype_s):
    src_type, etype, dst_type = canonical_etype_s.split(",")
    src_type = src_type[2:-1]
    dst_type = dst_type[2:-2]
    etype = etype[2:-1]
    return (src_type, etype, dst_type)


def create_dlpack_d(d):
    dlpack_d = {}
    for k, df in d.items():
        if len(df) == 0:
            dlpack_d[k] = (None, None, None)
        else:
            dlpack_d[k] = (
                df[src_n].to_dlpack(),
                df[dst_n].to_dlpack(),
                df[eid_n].to_dlpack(),
            )

    return dlpack_d


def get_underlying_dtype_from_sg(sg):
    """
    Returns the underlying dtype of the subgraph
    """
    # FIXME: Remove after we have consistent naming
    # https://github.com/rapidsai/cugraph/issues/2618
    sg_columns = sg.edgelist.edgelist_df.columns
    if "src" in sg_columns:
        # src for single node graph
        sg_node_dtype = sg.edgelist.edgelist_df["src"].dtype
    elif src_n in sg_columns:
        # _SRC_ for multi-node graphs
        sg_node_dtype = sg.edgelist.edgelist_df[src_n].dtype
    else:
        raise ValueError(f"Source column {src_n} not found in the subgraph")

    return sg_node_dtype
