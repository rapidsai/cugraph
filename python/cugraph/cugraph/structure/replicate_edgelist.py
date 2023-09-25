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


def replicate_edgelist(
    edgelist_ddf: dask_cudf.DataFrame = None,
) -> dask_cudf.DataFrame:
    """
    Select random vertices from the graph

    Parameters
    ----------


    Returns
    -------
    return
    """

    _client = default_client()

    edgelist_ddf = persist_dask_df_equal_parts_per_worker(edgelist_ddf, _client)
    edgelist_ddf = get_persisted_df_worker_map(edgelist_ddf, _client)

    def convert_to_cudf(cp_arrays: cp.ndarray) -> cudf.DataFrame:
        """
        Creates a cudf Dataframe from cupy arrays
        """
        # vertices = cudf.Series(cp_arrays)
        src, dst, wgt, _ = cp_arrays
        gathered_edgelist_df = cudf.DataFrame()
        gathered_edgelist_df["src"] = src
        gathered_edgelist_df["dst"] = dst
        gathered_edgelist_df["wgt"] = wgt
        # print("the gathered edgelist = \n", gathered_edgelist_df)

        return gathered_edgelist_df

    def _call_plc_replicate_edgelist(
        sID: bytes, edgelist_df: cudf.DataFrame
    ) -> cudf.Series:
        # print("edgelist_df = \n", edgelist_df)
        cp_arrays = pylibcugraph_replicate_edgelist(
            resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
            # FIXME: these are harcoded for now
            src_array=edgelist_df[0]["src"],
            dst_array=edgelist_df[0]["dst"],
            weight_array=edgelist_df[0][edgelist_df[0].columns[2]],
        )
        return convert_to_cudf(cp_arrays)

    def _mg_call_plc_replicate_edgelist(
        client: dask.distributed.client.Client,
        sID: bytes,
        edgelist_ddf: dict,
    ) -> dask_cudf.Series:

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
        # print("result before = \n", result)
        # wait(result)
        # print("result after = \n", result)
        ddf = dask_cudf.from_delayed(result, verify_meta=False).persist()
        wait(ddf)
        wait([r.release() for r in result])
        return ddf

    ddf = _mg_call_plc_replicate_edgelist(
        _client,
        Comms.get_session_id(),
        edgelist_ddf,
    )

    return ddf
