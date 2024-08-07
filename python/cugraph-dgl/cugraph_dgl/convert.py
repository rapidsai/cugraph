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
from cugraph.utilities.utils import import_optional

import cugraph_dgl
from cugraph_dgl import CuGraphStorage
from cugraph_dgl.utils.cugraph_conversion_utils import (
    get_edges_dict_from_dgl_HeteroGraph,
    add_ndata_from_dgl_HeteroGraph,
    add_edata_from_dgl_HeteroGraph,
)

dgl = import_optional("dgl")


def cugraph_storage_from_heterograph(
    g: dgl.DGLGraph, single_gpu: bool = True
) -> CuGraphStorage:
    """
    Convert DGL Graph to CuGraphStorage graph
    """
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    edges_dict = get_edges_dict_from_dgl_HeteroGraph(g, single_gpu)
    gs = CuGraphStorage(
        data_dict=edges_dict,
        num_nodes_dict=num_nodes_dict,
        single_gpu=single_gpu,
        idtype=g.idtype,
    )
    add_ndata_from_dgl_HeteroGraph(gs, g)
    add_edata_from_dgl_HeteroGraph(gs, g)
    return gs


def cugraph_dgl_graph_from_heterograph(
    input_graph: dgl.DGLGraph,
    single_gpu: bool = True,
    ndata_storage: str = "torch",
    edata_storage: str = "torch",
    **kwargs,
) -> cugraph_dgl.Graph:
    """
    Converts a DGL Graph to a cuGraph-DGL Graph.
    """

    output_graph = cugraph_dgl.Graph(
        is_multi_gpu=(not single_gpu),
        ndata_storage=ndata_storage,
        edata_storage=edata_storage,
        **kwargs,
    )

    # Calling is_homogeneous does not work here
    if len(input_graph.ntypes) <= 1:
        output_graph.add_nodes(
            input_graph.num_nodes(), data=input_graph.ndata, ntype=input_graph.ntypes[0]
        )
    else:
        for ntype in input_graph.ntypes:
            data = {
                k: v_dict[ntype]
                for k, v_dict in input_graph.ndata.items()
                if ntype in v_dict
            }
            output_graph.add_nodes(input_graph.num_nodes(ntype), data=data, ntype=ntype)

    if len(input_graph.canonical_etypes) <= 1:
        can_etype = input_graph.canonical_etypes[0]
        src_t, dst_t = input_graph.edges(form="uv", etype=can_etype)
        output_graph.add_edges(src_t, dst_t, input_graph.edata, etype=can_etype)
    else:
        for can_etype in input_graph.canonical_etypes:
            data = {
                k: v_dict[can_etype]
                for k, v_dict in input_graph.edata.items()
                if can_etype in v_dict
            }

            src_t, dst_t = input_graph.edges(form="uv", etype=can_etype)
            output_graph.add_edges(src_t, dst_t, data=data, etype=can_etype)

    return output_graph
