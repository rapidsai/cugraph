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

# Utils to convert b/w dgl heterograph to cugraph GraphStore
from __future__ import annotations
from typing import Dict, Tuple, Union

import cudf
import pandas as pd
import dask.dataframe as dd
import dask_cudf
from dask.distributed import get_client
import cupy as cp
from cugraph.utilities.utils import import_optional
from cugraph.gnn.dgl_extensions.dgl_uniform_sampler import src_n, dst_n

dgl = import_optional("dgl")
F = import_optional("dgl.backend")
torch = import_optional("torch")


# Feature Tensor to DataFrame Utils
def convert_to_column_major(t: torch.Tensor):
    return t.t().contiguous().t()


def create_ar_from_tensor(t: torch.Tensor):
    t = convert_to_column_major(t)
    if t.device.type == "cuda":
        ar = cp.asarray(t)
    else:
        ar = t.numpy()
    return ar


def _create_edge_frame(src_t: torch.Tensor, dst_t: torch.Tensor, single_gpu: bool):
    """
    Create a edge dataframe from src_t and dst_t
    """
    src_ar = create_ar_from_tensor(src_t)
    dst_ar = create_ar_from_tensor(dst_t)
    edge_df = _create_df_from_edge_ar(src_ar, dst_ar, single_gpu=single_gpu)
    edge_df = edge_df.rename(
        columns={edge_df.columns[0]: src_n, edge_df.columns[1]: dst_n}
    )
    return edge_df


def _create_df_from_edge_ar(src_ar, dst_ar, single_gpu=True):
    if not single_gpu:
        nworkers = len(get_client().scheduler_info()["workers"])
        npartitions = nworkers * 1
    if single_gpu:
        df = cudf.DataFrame(data={src_n: src_ar, dst_n: dst_ar})
    else:
        if isinstance(src_ar, cp.ndarray):
            src_ar = src_ar.get()
        if isinstance(dst_ar, cp.ndarray):
            dst_ar = dst_ar.get()

        df = pd.DataFrame(data={src_n: src_ar, dst_n: dst_ar})
        # Only save stuff in host memory
        df = dd.from_pandas(df, npartitions=npartitions).persist()
        df = df.map_partitions(cudf.DataFrame.from_pandas)

    df = df.reset_index(drop=True)
    return df


def get_edges_dict_from_dgl_HeteroGraph(
    graph: dgl.DGLHeteroGraph, single_gpu: bool
) -> Dict[Tuple[str, str, str], Union[cudf.DataFrame, dask_cudf.DataFrame]]:
    etype_d = {}
    for can_etype in graph.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        etype_d[can_etype] = _create_edge_frame(src_t, dst_t, single_gpu)
    return etype_d


def add_ndata_from_dgl_HeteroGraph(gs, g):
    for feat_name, feat in g.ndata.items():
        if isinstance(feat, torch.Tensor):
            assert len(g.ntypes) == 1
            ntype = g.ntypes[0]
            gs.ndata_storage.add_data(
                feat_name=feat_name, type_name=ntype, feat_obj=feat
            )
        else:
            for ntype, feat_t in feat.items():
                gs.ndata_storage.add_data(
                    feat_name=feat_name, type_name=ntype, feat_obj=feat_t
                )


def add_edata_from_dgl_HeteroGraph(gs, g):
    for feat_name, feat in g.edata.items():
        if isinstance(feat, torch.Tensor):
            assert len(g.etypes) == 1
            etype = g.etypes[0]
            gs.edata_storage.add_data(
                feat_name=feat_name, type_name=etype, feat_obj=feat
            )
        else:
            for etype, feat_t in feat.items():
                gs.edata_storage.add_data(
                    feat_name=feat_name, type_name=etype, feat_obj=feat_t
                )
