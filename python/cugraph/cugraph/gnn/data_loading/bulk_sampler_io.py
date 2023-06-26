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

import os
import cupy
import cudf
import dask_cudf

from dask import delayed
from dask.dataframe.utils import make_meta
from distributed import default_client
from cugraph.dask.common.part_utils import get_persisted_df_worker_map

from typing import Union, Optional, Sequence


def _write_samples_to_parquet(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    batches_per_partition: int,
    output_path: str,
    partition_info: Optional[Union[dict, str]] = None,
) -> None:
    """
    Writes the samples to parquet.
    results: cudf.DataFrame
        The results dataframe containing the sampled minibatches.
    offsets: cudf.DataFrame
        The offsets dataframe indicating the start/end of each minibatch
        in the reuslts dataframe.
    batches_per_partition: int
        The maximum number of minibatches allowed per written parquet partition.
    output_path: str
        The output path (where parquet files should be written to).
    partition_info: Union[dict, str]
        Either a dictionary containing partition data from dask, the string 'sg'
        indicating that this is a single GPU write, or None indicating that this
        function should perform a no-op (required by dask).
    """

    # Required by dask; need to skip dummy partitions.
    if partition_info is None or len(results) == 0:
        return
    if partition_info != "sg" and (not isinstance(partition_info, dict)):
        raise ValueError("Invalid value of partition_info")

    max_batch_id = offsets.batch_id.max()

    for p in range(0, len(offsets), batches_per_partition):
        offsets_p = offsets.iloc[p : p + batches_per_partition]
        start_batch_id = offsets_p.batch_id.iloc[0]
        end_batch_id = offsets_p.batch_id.iloc[-1]

        start_ix = offsets_p.offsets.iloc[0]
        if end_batch_id == max_batch_id:
            end_ix = len(results)
        else:
            end_ix = offsets.offsets[offsets.batch_id == (end_batch_id + 1)].iloc[0]

        full_output_path = os.path.join(
            output_path, f"batch={start_batch_id}-{end_batch_id}.parquet"
        )
        results_p = results.iloc[start_ix:end_ix]

        results_p["batch_id"] = offsets_p.batch_id.repeat(
            cupy.diff(offsets_p.offsets.values, append=end_ix)
        ).values
        results_p.to_parquet(full_output_path, compression=None, index=False)


def write_samples(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    batches_per_partition: cudf.DataFrame,
    output_path: str,
):
    """
    Writes the samples to parquet.
    results: cudf.DataFrame
        The results dataframe containing the sampled minibatches.
    offsets: cudf.DataFrame
        The offsets dataframe indicating the start/end of each minibatch
        in the reuslts dataframe.
    batches_per_partition: int
        The maximum number of minibatches allowed per written parquet partition.
    output_path: str
        The output path (where parquet files should be written to).
    """
    if hasattr(results, "compute"):
        results.map_partitions(
            _write_samples_to_parquet,
            offsets,
            batches_per_partition,
            output_path,
            align_dataframes=False,
        ).compute()
    else:
        _write_samples_to_parquet(
            results, offsets, batches_per_partition, output_path, partition_info="sg"
        )

def _filter_batches(
    batches: Sequence[cudf.DataFrame],
    batch_col_name: str,
    max_batch_id: int,
) -> cudf.DataFrame: 
    if isinstance(batches, cudf.DataFrame):
        batches = [batches]

    filtered_batches = cudf.DataFrame()
    next_batches = cudf.DataFrame()

    for df in batches:
        f = (df[batch_col_name] <= max_batch_id)
        filtered_batches = cudf.concat(
            [
                filtered_batches,
                df.loc[f]
            ],
            ignore_index=True
        )
        next_batches = cudf.concat(
            [
                next_batches,
                df.loc[~f]
            ]
        )
        for col in list(df.columns):
            df.drop(col, axis=1, inplace=True)
    
    return filtered_batches, next_batches

def filter_batches(
    batches:cudf.DataFrame,
    batch_col_name: str,
    max_batch_id: int
):
    if hasattr(batches, 'compute'):
        #old_len = len(batches)
        client = default_client()
        meta = make_meta(batches)

        batches = get_persisted_df_worker_map(batches, client)
        delayed_tasks_d = {
            w: delayed(_filter_batches)(
                bdata,
                batch_col_name,
                max_batch_id
            )
            for w, bdata in batches.items()
        }
        del batches

        result = [
            client.compute(
                task,
                workers=[w],
                allow_other_workers=False,
                pure=True,
            )
            for w, task in delayed_tasks_d.items()
        ]

        result = [delayed(lambda x: x, nout=2)(r) for r in result]
        filtered_batches = dask_cudf.from_delayed(
            [r[0] for r in result], meta=meta, verify_meta=False
        ).persist()
        batches = dask_cudf.from_delayed(
            [r[1] for r in result], meta=meta, verify_meta=False
        ).persist()

        #print(old_len)
        #print(len(batches))
        #print(len(filtered_batches))
        #assert len(filtered_batches) + len(batches) == old_len
    else:
        filtered_batches, batches = _filter_batches(
            batches,
            batch_col_name,
            max_batch_id,
            partition_info='sg',
        )
    
    return filtered_batches, batches