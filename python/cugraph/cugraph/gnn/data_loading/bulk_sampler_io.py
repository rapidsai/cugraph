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
import cudf
import cupy

from typing import Union, Optional


def _write_samples_to_parquet(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    batches_per_partition: int,
    output_path: str,
    partition_info: Optional[Union[dict, str]] = None,
) -> cudf.Series:
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

    Returns an empty cudf series.
    """

    # Required by dask; need to skip dummy partitions.
    if partition_info is None or len(results) == 0:
        return cudf.Series(dtype="int64")
    if partition_info != "sg" and (not isinstance(partition_info, dict)):
        raise ValueError("Invalid value of partition_info")

    max_batch_id = offsets.batch_id.max()
    results = results.dropna(axis=1, how='all')
    results["hop_id"]=results["hop_id"].astype('uint8')

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

    return cudf.Series(dtype="int64")


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
            meta=cudf.Series(dtype="int64"),
        ).compute()

    else:
        _write_samples_to_parquet(
            results, offsets, batches_per_partition, output_path, partition_info="sg"
        )
