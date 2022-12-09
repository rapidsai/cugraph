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
from cugraph.gnn.dgl_extensions.utils.sampling import (
    sample_pg,
    get_subgraph_and_src_range_from_pg,
)
from cugraph.gnn.dgl_extensions.utils.sampling import get_underlying_dtype_from_sg
import cupy as cp


def get_subgraph_and_src_range_from_pg_remote(graph_id, reverse_edges, etype, server):
    pG = server.get_graph(graph_id)
    subg, src_range = get_subgraph_and_src_range_from_pg(pG, reverse_edges, etype)
    g_id = server.add_graph(subg)
    g_id = cp.int8(g_id)
    return g_id, src_range


def get_underlying_dtype_from_sg_remote(graph_id, server):
    g = server.get_graph(graph_id)
    dtype_name = get_underlying_dtype_from_sg(g).name
    if dtype_name == "int32":
        return 32
    if dtype_name == "int64":
        return 64
    else:
        raise NotImplementedError(
            "IDS other than int32 and int64 not yet supported"
            f"got dtype = {dtype_name}"
        )


def sample_pg_remote(
    graph_id,
    has_multiple_etypes,
    etypes,
    sgs_obj,
    sgs_src_range_obj,
    sg_node_dtype,
    nodes_ar,
    replace,
    fanout,
    edge_dir,
    server,
):
    pg = server.get_graph(graph_id)

    if isinstance(sgs_obj, dict):
        sgs_obj = {k: server.get_graph(v) for k, v in sgs_obj.items()}
    else:
        sgs_obj = server.get_graph(sgs_obj)

    sampled_result_arrays = sample_pg(
        pg=pg,
        has_multiple_etypes=has_multiple_etypes,
        etypes=etypes,
        sgs_obj=sgs_obj,
        sgs_src_range_obj=sgs_src_range_obj,
        sg_node_dtype=sg_node_dtype,
        nodes_ar=nodes_ar,
        replace=replace,
        fanout=fanout,
        edge_dir=edge_dir,
    )

    return sampled_result_arrays
