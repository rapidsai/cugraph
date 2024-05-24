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
import numpy as np
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


# FIXME: Convert it to a general-purpose util function
def _convert_to_cudf(cp_arrays: Tuple[cp.ndarray], col_names: list) -> cudf.DataFrame:
    """
    Creates a cudf Dataframe from cupy arrays
    """
    src, dst, wgt, edge_id, edge_type_id, _ = cp_arrays
    gathered_edgelist_df = cudf.DataFrame()
    gathered_edgelist_df[col_names[0]] = src
    gathered_edgelist_df[col_names[1]] = dst
    if wgt is not None:
        gathered_edgelist_df[col_names[2]] = wgt
    if edge_id is not None:
        gathered_edgelist_df[col_names[3]] = edge_id
    if edge_type_id is not None:
        gathered_edgelist_df[col_names[4]] = edge_type_id

    return gathered_edgelist_df


def _call_plc_replicate_edgelist(
    sID: bytes, edgelist_df: cudf.DataFrame, col_names: list
) -> cudf.DataFrame:
    edgelist_df = edgelist_df[0]
    cp_arrays = pylibcugraph_replicate_edgelist(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        src_array=edgelist_df[col_names[0]],
        dst_array=edgelist_df[col_names[1]],
        weight_array=edgelist_df[col_names[2]] if len(col_names) > 2 else None,
        edge_id_array=edgelist_df[col_names[3]] if len(col_names) > 3 else None,
        edge_type_id_array=edgelist_df[col_names[4]] if len(col_names) > 4 else None,
    )
    return _convert_to_cudf(cp_arrays, col_names)


def _call_plc_replicate_dataframe(sID: bytes, df: cudf.DataFrame) -> cudf.DataFrame:
    df = df[0]
    df_replicated = cudf.DataFrame()
    for col_name in df.columns:
        cp_array = pylibcugraph_replicate_edgelist(
            resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
            src_array=df[col_name]
            if df[col_name].dtype in [np.int32, np.int64]
            else None,
            dst_array=None,
            weight_array=df[col_name]
            if df[col_name].dtype in [np.float32, np.float64]
            else None,
            edge_id_array=None,
            edge_type_id_array=None,
        )
        src, _, wgt, _, _, _ = cp_array
        if src is not None:
            df_replicated[col_name] = src
        elif wgt is not None:
            df_replicated[col_name] = wgt

    return df_replicated


def _call_plc_replicate_series(sID: bytes, series: cudf.Series) -> cudf.Series:
    series = series[0]
    series_replicated = cudf.Series()
    cp_array = pylibcugraph_replicate_edgelist(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        src_array=series if series.dtype in [np.int32, np.int64] else None,
        dst_array=None,
        weight_array=series if series.dtype in [np.float32, np.float64] else None,
        edge_id_array=None,
        edge_type_id_array=None,
    )
    src, _, wgt, _, _, _ = cp_array
    if src is not None:
        series_replicated = cudf.Series(src)
    elif wgt is not None:
        series_replicated = cudf.Series(wgt)

    return series_replicated


def _mg_call_plc_replicate(
    client: dask.distributed.client.Client,
    sID: bytes,
    dask_object: dict,
    input_type: str,
    col_names: list,
) -> Union[dask_cudf.DataFrame, dask_cudf.Series]:

    if input_type == "dataframe":
        result = [
            client.submit(
                _call_plc_replicate_dataframe,
                sID,
                edata,
                workers=[w],
                allow_other_workers=False,
                pure=False,
            )
            for w, edata in dask_object.items()
        ]
    elif input_type == "dataframe":
        result = [
            client.submit(
                _call_plc_replicate_series,
                sID,
                edata,
                workers=[w],
                allow_other_workers=False,
                pure=False,
            )
            for w, edata in dask_object.items()
        ]
    elif input_type == "edgelist":
        result = [
            client.submit(
                _call_plc_replicate_edgelist,
                sID,
                edata,
                col_names,
                workers=[w],
                allow_other_workers=False,
                pure=False,
            )
            for w, edata in dask_object.items()
        ]

    ddf = dask_cudf.from_delayed(result, verify_meta=False).persist()
    wait(ddf)
    wait([r.release() for r in result])
    return ddf


def replicate_edgelist(
    edgelist_ddf: Union[dask_cudf.DataFrame, cudf.DataFrame] = None,
    source="src",
    destination="dst",
    weight=None,
    edge_id=None,
    edge_type=None,
) -> dask_cudf.DataFrame:
    """
    Replicate edges across all GPUs

    Parameters
    ----------

    edgelist_ddf: cudf.DataFrame or dask_cudf.DataFrame
        A DataFrame that contains edge information.

    source : str or array-like
            source column name or array of column names

    destination : str or array-like
        destination column name or array of column names

    weight : str, optional (default=None)
        Name of the weight column in the input dataframe.

    edge_id : str, optional (default=None)
        Name of the edge id column in the input dataframe.

    edge_type : str, optional (default=None)
        Name of the edge type column in the input dataframe.

    Returns
    -------
    df : dask_cudf.DataFrame
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
    col_names = [source, destination]

    if weight is not None:
        col_names.append(weight)
    if edge_id is not None:
        col_names.append(edge_id)
    if edge_type is not None:
        col_names.append(edge_type)

    if not (set(col_names).issubset(set(edgelist_ddf.columns))):
        raise ValueError(
            "Invalid column names were provided: valid columns names are "
            f"{edgelist_ddf.columns}"
        )

    edgelist_ddf = persist_dask_df_equal_parts_per_worker(edgelist_ddf, _client)
    edgelist_ddf = get_persisted_df_worker_map(edgelist_ddf, _client)

    ddf = _mg_call_plc_replicate(
        _client,
        Comms.get_session_id(),
        edgelist_ddf,
        "edgelist",
        col_names,
    )

    return ddf


def replicate_cudf_dataframe(cudf_dataframe):
    """
    Replicate dataframe across all GPUs

    Parameters
    ----------

    cudf_dataframe: cudf.DataFrame or dask_cudf.DataFrame

    Returns
    -------
    df : dask_cudf.DataFrame
        A distributed dataframe where each partition contains the
        combined dataframe from all GPUs. If a cudf.DataFrame was passed
        as input, the dataframe will be replicated across all the other
        GPUs in the cluster. If as dask_cudf.DataFrame was passed as input,
        each partition will be filled with the datafame of all partitions
        in the dask_cudf.DataFrame.

    """

    supported_types = [np.int32, np.int64, np.float32, np.float64]
    if not all(dtype in supported_types for dtype in cudf_dataframe.dtypes):
        raise TypeError(
            "The supported types are 'int32', 'int64', 'float32', 'float64'"
        )

    _client = default_client()
    if not isinstance(cudf_dataframe, dask_cudf.DataFrame):
        if isinstance(cudf_dataframe, cudf.DataFrame):
            df = dask_cudf.from_cudf(
                cudf_dataframe, npartitions=len(Comms.get_workers())
            )
        elif not isinstance(cudf_dataframe, dask_cudf.DataFrame):
            raise TypeError(
                "The variable 'cudf_dataframe' must be of type "
                f"'cudf/dask_cudf.dataframe', got type {type(cudf_dataframe)}"
            )
    else:
        df = cudf_dataframe

    df = persist_dask_df_equal_parts_per_worker(df, _client)
    df = get_persisted_df_worker_map(df, _client)

    ddf = _mg_call_plc_replicate(
        _client,
        Comms.get_session_id(),
        df,
        "dataframe",
        cudf_dataframe.columns
    )

    return ddf


def replicate_cudf_series(cudf_series):
    """
    Replicate series across all GPUs

    Parameters
    ----------

    cudf_series: cudf.Series or dask_cudf.Series

    Returns
    -------
    series : dask_cudf.Series
        A distributed series where each partition contains the
        combined series from all GPUs. If a cudf.Series was passed
        as input, the Series will be replicated across all the other
        GPUs in the cluster. If as dask_cudf.Series was passed as input,
        each partition will be filled with the series of all partitions
        in the dask_cudf.Series.

    """

    supported_types = [np.int32, np.int64, np.float32, np.float64]
    if cudf_series.dtype not in supported_types:
        raise TypeError(
            "The supported types are 'int32', 'int64', 'float32', 'float64'"
        )

    _client = default_client()

    if not isinstance(cudf_series, dask_cudf.Series):
        if isinstance(cudf_series, cudf.Series):
            series = dask_cudf.from_cudf(
                cudf_series, npartitions=len(Comms.get_workers())
            )
        elif not isinstance(cudf_series, dask_cudf.Series):
            raise TypeError(
                "The variable 'cudf_series' must be of type "
                f"'cudf/dask_cudf.series', got type {type(cudf_series)}"
            )
    else:
        series = cudf_series

    series = persist_dask_df_equal_parts_per_worker(series, _client)
    series = get_persisted_df_worker_map(series, _client)

    series = _mg_call_plc_replicate(
        _client,
        Comms.get_session_id(),
        series,
        "series",
    )

    return series
