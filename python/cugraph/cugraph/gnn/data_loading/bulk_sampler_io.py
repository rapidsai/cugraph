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
from cugraph.utilities.utils import import_optional

from typing import Union, Optional

dgl = import_optional("dgl")
torch = import_optional("torch")


def _get_renumbered_s_d(s, d, sampling_direction="in"):
    # We have to make
    # reverse graph in dgl
    # then renumber
    # for downstream compatibility
    if sampling_direction == "in":
        s, d = d, s

    sampled_graph = dgl.graph((s, d))
    block = dgl.to_block(
        sampled_graph,
        dst_nodes=d.unique(),
        src_nodes=s.unique(),
        include_dst_in_src=True,
    )
    rs, rd = block.adj_tensors("coo")

    if sampling_direction == "in":
        rd, rs = rs, rd
    return rs, rd


def _write_samples_to_parquet(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    batches_per_partition: int,
    output_path: str,
    renumber_using_dgl: bool = False,
    sampling_direction: str = "in",
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

    if renumber_using_dgl:
        print("Renumbering using dgl")
        print("WARNING: This is a temporary test for downstream compatibility")

    # Remove empty columns
    results = results.dropna(axis=1, how="all")
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

        if renumber_using_dgl:
            results_p["renumbered_sources"] = cupy.int32(-1)
            results_p["renumbered_destinations"] = cupy.int32(-1)
            hop_offsets = (
                results_p.index[results_p["hop_id"].diff() != 0].to_arrow().to_pylist()
            )
            hop_offsets.append(len(results))
            hop_start_ix = 0
            for hop_end_ix in hop_offsets:
                hop_s = results_p["sources"].iloc[hop_start_ix:hop_end_ix].values
                hop_d = results_p["destinations"].iloc[hop_start_ix:hop_end_ix].values
                hop_s = torch.tensor(hop_s, device="cuda")
                hop_d = torch.tensor(hop_d, device="cuda")
                hop_rs, hop_rd = _get_renumbered_s_d(
                    hop_s, hop_d, sampling_direction=sampling_direction
                )
                results_p["renumbered_sources"].iloc[
                    hop_start_ix:hop_end_ix
                ] = cupy.asarray(hop_rs)
                results_p["renumbered_destinations"].iloc[
                    hop_start_ix:hop_end_ix
                ] = cupy.asarray(hop_rd)
                hop_start_ix = hop_end_ix

        results_p["batch_id"] = offsets_p.batch_id.repeat(
            cupy.diff(offsets_p.offsets.values, append=end_ix)
        ).values

        results_p.to_parquet(full_output_path, compression=None, index=False)


def write_samples(
    results: cudf.DataFrame,
    offsets: cudf.DataFrame,
    batches_per_partition: cudf.DataFrame,
    output_path: str,
    renumber_using_dgl: bool = False,
    sampling_direction: str = "in",
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
            renumber_using_dgl=renumber_using_dgl,
            sampling_direction=sampling_direction,
            align_dataframes=False,
        ).compute()
    else:
        _write_samples_to_parquet(
            results,
            offsets,
            batches_per_partition,
            output_path,
            renumber_using_dgl,
            sampling_direction,
            partition_info="sg",
        )
