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

import dask_cudf
import cudf
from dask.distributed import wait, default_client
from pylibcugraph import (
    ResourceHandle,
    replicate_edgelist as pylibcugraph_replicate_edgelist,
)

from cugraph.dask.common.part_utils import (
    get_persisted_df_worker_map,
    persist_dask_df_equal_parts_per_worker,
)

import dask
import cupy as cp
import cugraph.dask.comms.comms as Comms
from typing import Union, Tuple

edgeWeightCol = "weights"
edgeIdCol = "edge_id"
edgeTypeCol = "edge_type"
srcCol = "src"
dstCol = "dst"


def convert_to_cudf(cp_arrays: Tuple[cp.ndarray]) -> cudf.DataFrame:
        """
        Creates a cudf Dataframe from cupy arrays
        """
        src, dst, wgt, edge_id, edge_type_id, _ = cp_arrays
        gathered_edgelist_df = cudf.DataFrame()
        gathered_edgelist_df[srcCol] = src
        gathered_edgelist_df[dstCol] = dst
        if wgt is not None:
            gathered_edgelist_df[edgeWeightCol] = wgt
        if edge_id is not None:
            gathered_edgelist_df[edgeIdCol] = edge_id
        if edge_type_id is not None:
            gathered_edgelist_df[edgeTypeCol] = edge_type_id

        return gathered_edgelist_df

def _call_plc_replicate_edgelist(
        sID: bytes, edgelist_df: cudf.DataFrame
    ) -> cudf.DataFrame:
        edgelist_df = edgelist_df[0]
        cp_arrays = pylibcugraph_replicate_edgelist(
            resource_handle=ResourceHandle(
                Comms.get_handle(sID).getHandle()),
            src_array=edgelist_df[srcCol],
            dst_array=edgelist_df[dstCol],
            weight_array=edgelist_df[edgeWeightCol] \
                if edgeWeightCol in edgelist_df.columns else None,
            edge_id_array=edgelist_df[edgeIdCol] \
                if edgeIdCol in edgelist_df.columns else None,
            edge_type_id_array=edgelist_df[edgeTypeCol] \
                if edgeTypeCol in edgelist_df.columns else None,
        )
        return convert_to_cudf(cp_arrays)

def _mg_call_plc_replicate_edgelist(
        client: dask.distributed.client.Client,
        sID: bytes,
        edgelist_ddf: dict,
    ) -> dask_cudf.DataFrame:

        result = [
            client.submit(
                _call_plc_replicate_edgelist,
                sID,
                edata,
                workers=[w],
                allow_other_workers=False,
                pure=False,
            )
            for w, edata in edgelist_ddf.items()
        ]
        ddf = dask_cudf.from_delayed(result, verify_meta=False).persist()
        wait(ddf)
        wait([r.release() for r in result])
        return ddf


def replicate_edgelist(
    edgelist_ddf: Union[dask_cudf.DataFrame, cudf.DataFrame] = None,
) -> dask_cudf.DataFrame:
    """
    Replicate edges across all GPUs

    Parameters
    ----------

    edgelist_ddf: cudf.DataFrame or dask_cudf.DataFrame
        A DataFrame that contains edge information.

    Returns
    -------
    df : cudf.DataFrame or dask_cudf.DataFrame
        A distributed dataframe where each partition contains the
        combined edgelist from all GPUs. If a cudf.DataFrame was passed
        as input, the edgelist will be replicated across all the other
        GPUs in the cluster. If as dask_cudf.DataFrame was passed as input,
        each partition will be filled with the edges of all partitions
        in the dask_cudf.DataFrame.

    """

    _client = default_client()

    if isinstance(edgelist_ddf, cudf.DataFrame):
        edgelist_ddf = dask_cudf.from_cudf(
            edgelist_ddf, npartitions=len(Comms.get_workers())
        )

    valid_columns = [edgeWeightCol, edgeIdCol, edgeTypeCol, srcCol, dstCol]

    if not (set(edgelist_ddf.columns).issubset(set(valid_columns))):
        raise ValueError(
            "Invalid column names were provided: valid columns names are "
            f"{srcCol}, {dstCol}, {edgeWeightCol}, {edgeIdCol} "
            f"and {edgeTypeCol}"
        )

    edgelist_ddf = persist_dask_df_equal_parts_per_worker(
        edgelist_ddf, _client)
    edgelist_ddf = get_persisted_df_worker_map(edgelist_ddf, _client)

    ddf = _mg_call_plc_replicate_edgelist(
        _client,
        Comms.get_session_id(),
        edgelist_ddf,
    )

    return ddf
