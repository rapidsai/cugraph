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

from math import ceil

from pandas import isna

from typing import Union, Optional, List


def create_df_from_disjoint_series(series_list: List[cudf.Series]):
    series_list.sort(key=lambda s: len(s), reverse=True)

    df = cudf.DataFrame()
    for s in series_list:
        df[s.name] = s

    return df


def _write_samples_to_parquet_csr(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    renumber_map: cudf.DataFrame,
    batches_per_partition: int,
    output_path: str,
    partition_info: Optional[Union[dict, str]] = None,
) -> cudf.Series:
    """
    Writes CSR/CSC compressed samples to parquet.

    Batches that are empty are discarded, and the remaining non-empty
    batches are renumbered to be contiguous starting from the first
    batch id.  This means that the output batch ids may not match
    the input batch ids.

    results: cudf.DataFrame
        The results dataframe containing the sampled minibatches.
    offsets: cudf.DataFrame
        The offsets dataframe indicating the start/end of each minibatch
        in the reuslts dataframe.
    renumber_map: cudf.DataFrame
        The renumber map containing the mapping of renumbered vertex ids
        to original vertex ids.
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

    # Additional check to skip dummy partitions required for CSR format.
    if isna(offsets.batch_id.iloc[0]):
        return cudf.Series(dtype="int64")

    # Output:
    # major_offsets - CSR/CSC row/col pointers
    # minors - CSR/CSC col/row indices
    # edge id - edge ids (same shape as minors)
    # edge type - edge types (same shape as minors)
    # weight - edge weight (same shape as minors)
    # renumber map - the original vertex ids
    # renumber map offsets - start/end of the map for each batch
    #                        (only 1 per batch b/c of framework
    #                         stipulations making this legal)
    # label-hop offsets - indicate the start/end of each hop
    #                     for each batch

    batch_ids = offsets.batch_id
    label_hop_offsets = offsets.offsets
    renumber_map_offsets = offsets.renumber_map_offsets
    del offsets

    batch_ids.dropna(inplace=True)
    label_hop_offsets.dropna(inplace=True)
    renumber_map_offsets.dropna(inplace=True)

    major_offsets_array = results.major_offsets
    results.drop(columns="major_offsets", inplace=True)
    major_offsets_array.dropna(inplace=True)
    major_offsets_array = major_offsets_array.values

    minors_array = results.minors
    results.drop(columns="minors", inplace=True)
    minors_array.dropna(inplace=True)
    minors_array = minors_array.values

    weight_array = results.weight
    results.drop(columns="weight", inplace=True)
    weight_array.dropna(inplace=True)
    weight_array = (
        cupy.array([], dtype="float32") if weight_array.empty else weight_array.values
    )

    edge_id_array = results.edge_id
    results.drop(columns="edge_id", inplace=True)
    edge_id_array.dropna(inplace=True)
    edge_id_array = (
        cupy.array([], dtype="int64") if edge_id_array.empty else edge_id_array.values
    )

    edge_type_array = results.edge_type
    results.drop(columns="edge_type", inplace=True)
    edge_type_array.dropna(inplace=True)
    edge_type_array = (
        cupy.array([], dtype="int32")
        if edge_type_array.empty
        else edge_type_array.values
    )

    del results

    offsets_length = len(label_hop_offsets) - 1
    if offsets_length % len(batch_ids) != 0:
        raise ValueError("Invalid hop offsets")
    fanout_length = int(offsets_length / len(batch_ids))

    for p in range(0, int(ceil(len(batch_ids) / batches_per_partition))):
        partition_start = p * (batches_per_partition)
        partition_end = (p + 1) * (batches_per_partition)

        label_hop_offsets_current_partition = label_hop_offsets.iloc[
            partition_start * fanout_length : partition_end * fanout_length + 1
        ].reset_index(drop=True)
        label_hop_offsets_current_partition.name = "label_hop_offsets"

        batch_ids_current_partition = batch_ids.iloc[partition_start:partition_end]

        (
            major_offsets_start,
            major_offsets_end,
        ) = label_hop_offsets_current_partition.iloc[
            [0, -1]
        ].values  # legal since offsets has the 1 extra offset
        results_start, results_end = major_offsets_array[
            [major_offsets_start, major_offsets_end]
        ]  # avoid d2h copy

        # no need to use end batch id, just ensure the batch is labeled correctly
        start_batch_id = batch_ids_current_partition.iloc[0]
        # end_batch_id = batch_ids_current_partition.iloc[-1]

        # create the renumber map offsets
        renumber_map_offsets_current_partition = renumber_map_offsets.iloc[
            partition_start : partition_end + 1
        ].reset_index(drop=True)
        renumber_map_offsets_current_partition.name = "renumber_map_offsets"

        (
            renumber_map_start,
            renumber_map_end,
        ) = renumber_map_offsets_current_partition.iloc[
            [0, -1]
        ].values  # avoid d2h copy

        results_current_partition = create_df_from_disjoint_series(
            [
                cudf.Series(minors_array[results_start:results_end], name="minors"),
                cudf.Series(
                    renumber_map.map.values[renumber_map_start:renumber_map_end],
                    name="map",
                ),
                label_hop_offsets_current_partition,
                cudf.Series(
                    major_offsets_array[major_offsets_start : major_offsets_end + 1],
                    name="major_offsets",
                ),
                cudf.Series(weight_array[results_start:results_end], name="weight"),
                cudf.Series(edge_id_array[results_start:results_end], name="edge_id"),
                cudf.Series(
                    edge_type_array[results_start:results_end], name="edge_type"
                ),
                renumber_map_offsets_current_partition,
            ]
        )

        end_batch_id = start_batch_id + len(batch_ids_current_partition) - 1
        filename = f"batch={start_batch_id}-{end_batch_id}.parquet"
        full_output_path = os.path.join(output_path, filename)

        results_current_partition.to_parquet(
            full_output_path, compression=None, index=False, force_nullable_schema=True
        )

    return cudf.Series(dtype="int64")


def _write_samples_to_parquet_coo(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    renumber_map: cudf.DataFrame,
    batches_per_partition: int,
    output_path: str,
    partition_info: Optional[Union[dict, str]] = None,
) -> cudf.Series:
    """
    Writes COO compressed samples to parquet.

    Batches that are empty are discarded, and the remaining non-empty
    batches are renumbered to be contiguous starting from the first
    batch id.  This means that the output batch ids may not match
    the input batch ids.

    results: cudf.DataFrame
        The results dataframe containing the sampled minibatches.
    offsets: cudf.DataFrame
        The offsets dataframe indicating the start/end of each minibatch
        in the reuslts dataframe.
    renumber_map: cudf.DataFrame
        The renumber map containing the mapping of renumbered vertex ids
        to original vertex ids.
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

    offsets = offsets[:-1]

    # Offsets is always in order, so the last batch id is always the highest
    max_batch_id = offsets.batch_id.iloc[-1]
    results.dropna(axis=1, how="all", inplace=True)
    results["hop_id"] = results["hop_id"].astype("uint8")

    for p in range(0, len(offsets), batches_per_partition):
        offsets_p = offsets.iloc[p : p + batches_per_partition]
        start_batch_id = offsets_p.batch_id.iloc[0]
        end_batch_id = offsets_p.batch_id.iloc[len(offsets_p) - 1]

        reached_end = end_batch_id == max_batch_id

        start_ix = offsets_p.offsets.iloc[0]
        if reached_end:
            end_ix = len(results)
        else:
            offsets_z = offsets[offsets.batch_id == (end_batch_id + 1)]
            end_ix = offsets_z.offsets.iloc[0]

        results_p = results.iloc[start_ix:end_ix].reset_index(drop=True)

        if end_batch_id - start_batch_id + 1 > len(offsets_p):
            # This occurs when some batches returned 0 samples.
            # To properly account this, the remaining batches are
            # renumbered to have contiguous batch ids and the empty
            # samples are dropped.
            offsets_p.drop("batch_id", axis=1, inplace=True)
            batch_id_range = cudf.Series(
                cupy.arange(start_batch_id, start_batch_id + len(offsets_p))
            )
            end_batch_id = start_batch_id + len(offsets_p) - 1
        else:
            batch_id_range = offsets_p.batch_id

        results_p["batch_id"] = batch_id_range.repeat(
            cupy.diff(offsets_p.offsets.values, append=end_ix)
        ).values

        if renumber_map is not None:
            renumber_map_start_ix = offsets_p.renumber_map_offsets.iloc[0]

            if reached_end:
                renumber_map_end_ix = len(renumber_map)
            else:
                renumber_map_end_ix = offsets_z.renumber_map_offsets.iloc[0]

            renumber_map_p = renumber_map.map.iloc[
                renumber_map_start_ix:renumber_map_end_ix
            ]

            # Add the length so no na-checking is required in the loading stage
            map_offset = (
                end_batch_id - start_batch_id + 2
            ) - offsets_p.renumber_map_offsets.iloc[0]
            renumber_map_o = cudf.concat(
                [
                    offsets_p.renumber_map_offsets + map_offset,
                    cudf.Series(
                        [len(renumber_map_p) + len(offsets_p) + 1], dtype="int32"
                    ),
                ]
            )

            renumber_offset_len = len(renumber_map_o)
            if renumber_offset_len != end_batch_id - start_batch_id + 2:
                raise ValueError("Invalid batch id or renumber map")

            final_map_series = cudf.concat(
                [
                    renumber_map_o,
                    renumber_map_p,
                ],
                ignore_index=True,
            )

            if len(final_map_series) > len(results_p):
                # this should rarely happen and only occurs on small graphs/samples
                # TODO remove the sort_index to improve performance on small graphs
                final_map_series.name = "map"
                results_p = results_p.join(final_map_series, how="outer").sort_index()
            else:
                results_p["map"] = final_map_series

        full_output_path = os.path.join(
            output_path, f"batch={start_batch_id}-{end_batch_id}.parquet"
        )

        results_p.to_parquet(
            full_output_path, compression=None, index=False, force_nullable_schema=True
        )

    return cudf.Series(dtype="int64")


def write_samples(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    renumber_map: cudf.DataFrame,
    batches_per_partition: cudf.DataFrame,
    output_path: str,
):
    """
    Writes the samples to parquet.

    Batches in each partition that are empty are discarded, and the remaining non-empty
    batches are renumbered to be contiguous starting from the first
    batch id in the partition.
    This means that the output batch ids may not match the input batch ids.

    results: cudf.DataFrame
        The results dataframe containing the sampled minibatches.
    offsets: cudf.DataFrame
        The offsets dataframe indicating the start/end of each minibatch
        in the reuslts dataframe.
    renumber_map: cudf.DataFrame
        The renumber map containing the mapping of renumbered vertex ids
        to original vertex ids.
    batches_per_partition: int
        The maximum number of minibatches allowed per written parquet partition.
    output_path: str
        The output path (where parquet files should be written to).
    """

    if ("majors" in results.columns) and ("minors" in results.columns):
        write_fn = _write_samples_to_parquet_coo

    # TODO these names will be deprecated in release 23.12
    elif ("sources" in results.columns) and ("destinations" in results.columns):
        write_fn = _write_samples_to_parquet_coo

    elif "major_offsets" in results.columns and "minors" in results.columns:
        write_fn = _write_samples_to_parquet_csr

    else:
        raise ValueError("invalid columns")

    if hasattr(results, "compute"):
        results.map_partitions(
            write_fn,
            offsets,
            renumber_map,
            batches_per_partition,
            output_path,
            align_dataframes=False,
            meta=cudf.Series(dtype="int64"),
        ).compute()

    else:
        write_fn(
            results,
            offsets,
            renumber_map,
            batches_per_partition,
            output_path,
            partition_info="sg",
        )
