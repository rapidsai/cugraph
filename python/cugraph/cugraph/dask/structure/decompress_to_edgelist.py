# Copyright (c) 2022-2024, NVIDIA CORPORATION.
#
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
#

from dask.distributed import wait, default_client

import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import cupy as cp

from typing import Tuple

from pylibcugraph import (
    ResourceHandle,
    decompress_to_edgelist as pylibcugraph_decompress_to_edgelist
)


def _call_decompress_to_edgelist(
    sID: bytes,
    mg_graph_x,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    return pylibcugraph_decompress_to_edgelist(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays: cp.ndarray) -> cudf.DataFrame:
    cp_src, cp_dst, cp_weight, cp_edge_ids, cp_edge_type_ids = cp_arrays

    df = cudf.DataFrame()
    df["src"] = cp_src
    df["dst"] = cp_dst
    if cp_weight is not None:
        df["weight"] = cp_weight
    if cp_edge_ids is not None:
        df["edge_ids"] = cp_edge_ids
    if cp_edge_type_ids is not None:
        df["edge_type_ids"] = cp_edge_type_ids

    return df


def decompress_to_edgelist(
    input_graph
) -> dask_cudf.DataFrame:
    """
    Extract a the edgelist from a graph.

    Parameters
    ----------
    G : cugraph.Graph or networkx.Graph
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    Returns
    -------
    ego_edge_lists : dask_cudf.DataFrame
        Distributed GPU data frame containing all induced sources identifiers,
        destination identifiers, and if applicable edge weights, edge ids and
        edge types
    """

    # Initialize dask client
    client = default_client()

    do_expensive_check = False

    result = [
        client.submit(
            _call_decompress_to_edgelist,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            do_expensive_check
        )
        for w in Comms.get_workers()
    ]
    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "src")
        ddf = input_graph.unrenumber(ddf, "dst")

    return ddf
