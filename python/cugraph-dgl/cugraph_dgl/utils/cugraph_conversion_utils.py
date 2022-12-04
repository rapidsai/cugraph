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

# Utils to convert b/w dgl heterograph to cugraph GraphStore
from __future__ import annotations
from typing import Optional, Dict, Union
import numpy as np
import cudf
import dask_cudf
from cugraph.utilities.utils import import_optional, MissingModule
from cugraph_dgl.utils.utils import create_ar_from_tensor, create_df_from_ar
from cugraph_dgl.utils.cugraph_storage_utils import backend_dtype_to_np_dtype_dict
import cugraph_dgl

dgl = import_optional("dgl")
F = import_optional("dgl.backend")
torch = import_optional("torch")


def _create_edge_frame(src_t: torch.Tensor, dst_t: torch.Tensor, single_gpu: bool):
    """
    Create a edge dataframe from src_t and dst_t
    """
    src_ar = create_ar_from_tensor(src_t)
    dst_ar = create_ar_from_tensor(dst_t)
    edge_ar = np.stack([src_ar, dst_ar], axis=1)
    edge_df, edge_columns = create_df_from_ar(edge_ar, single_gpu=single_gpu)
    edge_df = edge_df.rename(columns={edge_columns[0]: "src", edge_columns[1]: "dst"})
    return edge_df


def _create_feature_frame(
    feat_t_d: Dict[str, torch.Tensor],
    single_gpu: bool = True,
    frame_type: str = "node",
    idtype=None if isinstance(F, MissingModule) else F.int64,
    src_t: Optional[torch.Tensor] = None,
    dst_t: Optional[torch.Tensor] = None,
) -> Union[cudf.DataFrame, dask_cudf.DataFrame]:
    """
    Convert a feature_tensor_d to a dataframe
    """
    df_ls = []
    feat_name_map = {}
    for feat_key, feat_t in feat_t_d.items():
        feat_ar = create_ar_from_tensor(feat_t)
        del feat_t
        df, feat_columns = create_df_from_ar(feat_ar, feat_key, single_gpu=single_gpu)
        feat_name_map[feat_key] = feat_columns
        df_ls.append(df)

    if single_gpu:
        df = cudf.concat(df_ls, axis=1)
    else:
        df = dask_cudf.concat(df_ls, axis=1)

    if frame_type == "node":
        # Append node_ids to the node dataframe
        np_dtype = backend_dtype_to_np_dtype_dict[idtype]
        df["node_id"] = np_dtype(1)
        df["node_id"] = df["node_id"].cumsum() - 1
    else:
        # Append edges to the feature dataframe
        edge_df = _create_edge_frame(src_t, dst_t, single_gpu)
        if single_gpu:
            df = cudf.concat([df, edge_df], axis=1)
        else:
            df = dask_cudf.concat([df, edge_df], axis=1)

    return df, feat_name_map


# Add ndata utils
def _add_ndata_of_single_type(
    gs: cugraph_dgl.CuGraphStorage,
    feat_t_d: Optional[Dict[torch.Tensor]],
    ntype: str,
    idtype=None if isinstance(F, MissingModule) else F.int64,
):

    df, feat_name_map = _create_feature_frame(
        feat_t_d,
        single_gpu=gs.single_gpu,
        frame_type="node",
        idtype=idtype,
    )

    gs.add_node_data(
        df,
        "node_id",
        ntype=ntype,
        feat_name=feat_name_map,
        contains_vector_features=True,
    )
    return gs


def add_nodes_from_dgl_HeteroGraph(
    gs: cugraph_dgl.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
):
    if len(graph.ntypes) > 1:
        ntype_feat_d = dict()
        for feat_name in graph.ndata.keys():
            for ntype in graph.ndata[feat_name]:
                if ntype not in ntype_feat_d:
                    ntype_feat_d[ntype] = {}
                ntype_feat_d[ntype][feat_name] = graph.ndata[feat_name][ntype]

        for ntype in gs.num_nodes_dict.keys():
            feat_t_d = ntype_feat_d.get(ntype, None)
            if feat_t_d is not None:
                gs = _add_ndata_of_single_type(
                    gs=gs,
                    feat_t_d=feat_t_d,
                    ntype=ntype,
                    idtype=graph.idtype,
                )
    else:
        ntype = graph.ntypes[0]
        if graph.ndata:
            gs = _add_ndata_of_single_type(
                gs,
                feat_t_d=graph.ndata,
                ntype=ntype,
                idtype=graph.idtype,
            )
    return gs


# Add edata utils
def _add_edata_of_single_type(
    gs: cugraph_dgl.CuGraphStorage,
    feat_t_d: Optional[Dict[torch.Tensor]],
    src_t: torch.Tensor,
    dst_t: torch.Tensor,
    can_etype: tuple([str, str, str]),
):

    if feat_t_d:
        df, feat_name_map = _create_feature_frame(
            feat_t_d,
            single_gpu=gs.single_gpu,
            frame_type="edge",
            src_t=src_t,
            dst_t=dst_t,
        )
        feat_name = feat_name_map
        contains_vector_features = True
    else:
        df = _create_edge_frame(src_t, dst_t, gs.single_gpu)
        feat_name = None
        contains_vector_features = False

    gs.add_edge_data(
        df,
        ["src", "dst"],
        canonical_etype=can_etype,
        feat_name=feat_name,
        contains_vector_features=contains_vector_features,
    )
    return gs


def add_edges_from_dgl_HeteroGraph(
    gs: cugraph_dgl.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
):
    etype_feat_d = dict()
    for feat_name in graph.edata.keys():
        for etype in graph.edata[feat_name].keys():
            if etype not in etype_feat_d:
                etype_feat_d[etype] = {}
            etype_feat_d[etype][feat_name] = graph.edata[feat_name][etype]

    for can_etype in graph.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        feat_t_d = etype_feat_d.get(can_etype, None)
        _add_edata_of_single_type(gs, feat_t_d, src_t, dst_t, can_etype)
