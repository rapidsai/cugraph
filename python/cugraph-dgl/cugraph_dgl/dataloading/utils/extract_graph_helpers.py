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
from __future__ import annotations
from typing import Tuple, Dict, Union
import cugraph
import cudf
import dask_cudf
import numpy as np


def create_cugraph_graph_from_edges_dict(
    edges_dict: Dict[Tuple(str, str, str), Union[dask_cudf.DataFrame, cudf.DataFrame]],
    etype_id_dict: Dict[Dict[Tuple(str, str, str)] : int],
    edge_dir: str,
):
    if edge_dir == "in":
        edges_dict = {k: reverse_edges(df) for k, df in edges_dict.items()}
    # TODO: Potentially skip this for memory efficiencies
    edges_dict = {k: add_etype_id(df, etype_id_dict[k]) for k, df in edges_dict.items()}

    edges_dfs = list(edges_dict.values())
    del edges_dict
    if isinstance(edges_dfs[0], dask_cudf.DataFrame):
        edges_df = dask_cudf.concat(edges_dfs, ignore_index=True)
    else:
        edges_df = cudf.concat(edges_dfs, ignore_index=True)
    del edges_dfs

    edges_df["wgt"] = np.float32(0)
    G = cugraph.MultiGraph(directed=True)
    if isinstance(edges_df, dask_cudf.DataFrame):
        G.from_dask_cudf_edgelist(
            edges_df,
            source="_SRC_",
            destination="_DST_",
            edge_attr=["wgt", "_EDGE_ID_", "etp"],
            legacy_renum_only=True,
            renumber=True,
        )
    else:
        G.from_cudf_edgelist(
            edges_df,
            source="_SRC_",
            destination="_DST_",
            edge_attr=["wgt", "_EDGE_ID_", "etp"],
            legacy_renum_only=True,
            renumber=True,
        )
    return G


def reverse_edges(df: Union[dask_cudf.DataFrame, cudf.DataFrame]):
    return df.rename(columns={"_SRC_": "_DST_", "_DST_": "_SRC_"})


def add_etype_id(df: Union[dask_cudf.DataFrame, cudf.DataFrame], etype_id: int):
    df["etp"] = np.int32(etype_id)
    return df
