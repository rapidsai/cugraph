import os
import cudf
import cupy

from typing import Union, Optional

def _write_samples_to_parquet(results: cudf.DataFrame,
                               offsets:cudf.DataFrame,
                               batches_per_partition:int,
                               output_path:str,
                               partition_info:Optional[Union[dict, str]]=None) -> None:
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
    if partition_info is None:
        return
    if partition_info != 'sg' and (not isinstance(partition_info, dict)):
        raise ValueError('Invalid value of partition_info')

    max_batch_id = offsets.batch_id.max()

    for p in range(0, len(offsets), batches_per_partition):
        offsets_p = offsets.iloc[p:p+batches_per_partition]
        start_batch_id = offsets_p.batch_id.iloc[0]
        end_batch_id = offsets_p.batch_id.iloc[-1]

        start_ix = offsets_p.offsets.iloc[0]
        if end_batch_id == max_batch_id:
            end_ix = len(results)
        else:
            end_ix = offsets.offsets[offsets.batch_id==(end_batch_id+1)].iloc[0]
        
        full_output_path = os.path.join(output_path, f'batch={start_batch_id}-{end_batch_id}.parquet')
        results_p = results.iloc[start_ix:end_ix]

        results_p['batch_id'] = offsets_p.batch_id.repeat(cupy.diff(offsets_p.offsets.values, append=end_ix)).values
        results_p.to_parquet(full_output_path)

def write_samples(results: cudf.DataFrame,
                  offsets: cudf.DataFrame,
                  batches_per_partition: cudf.DataFrame,
                  output_path: str):
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
            align_dataframes=False
        ).compute()
    else:
        _write_samples_to_parquet(
            results,
            offsets,
            batches_per_partition,
            output_path,
            partition_info='sg'
        )