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

from typing import Union

import cudf
import dask_cudf

from dask.distributed import wait
from dask.distributed import futures_of

import cugraph
import pylibcugraph

from cugraph.gnn.data_loading.bulk_sampler_io import write_samples

import warnings
import logging
import time


class EXPERIMENTAL__BulkSampler:
    start_col_name = "_START_"
    batch_col_name = "_BATCH_"

    def __init__(
        self,
        batch_size: int,
        output_path: str,
        graph,
        seeds_per_call: int = 200_000,
        batches_per_partition: int = 100,
        log_level: int = None,
        **kwargs,
    ):
        """
        Constructs a new BulkSampler

        Parameters
        ----------
        batch_size: int
            The size of each batch.
        output_path: str
            The directory where results will be stored.
        graph: cugraph.Graph
            The cugraph graph to operate upon.
        seeds_per_call: int (optional, default=200,000)
            The number of seeds (start vertices) that can be processed by
            a single sampling call.
        batches_per_partition: int (optional, default=100)
            The number of batches outputted to a single parquet partition.
        log_level: int (optional, default=None)
            Whether to enable logging for this sampler. Supports 3 levels
            of logging if enabled (INFO, WARNING, ERROR).  If not provided,
            defaults to WARNING.
        kwargs: kwargs
            Keyword arguments to be passed to the sampler (i.e. fanout).
        """

        self.__logger = logging.getLogger(__name__)
        self.__logger.setLevel(log_level or logging.WARNING)

        max_batches_per_partition = seeds_per_call // batch_size
        if batches_per_partition > max_batches_per_partition:

            warnings.warn(
                f"batches_per_partition ({batches_per_partition}) is >"
                f" seeds_per_call / batch size ({max_batches_per_partition})"
                "; automatically setting batches_per_partition to "
                "{max_batches_per_partition}"
            )
            batches_per_partition = max_batches_per_partition

        self.__batch_size = batch_size
        self.__output_path = output_path
        self.__graph = graph
        self.__seeds_per_call = seeds_per_call
        self.__batches_per_partition = batches_per_partition
        self.__batches = None
        self.__sample_call_args = kwargs

    @property
    def seeds_per_call(self) -> int:
        return self.__seeds_per_call

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @property
    def batches_per_partition(self) -> int:
        return self.__batches_per_partition

    @property
    def size(self) -> int:
        if self.__batches is None:
            return 0
        else:
            return len(self.__batches)

    def add_batches(
        self,
        df: Union[cudf.DataFrame, dask_cudf.DataFrame],
        start_col_name: str,
        batch_col_name: str,
    ) -> None:
        """
        Adds batches to this BulkSampler.

        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            Contains columns for vertex ids, batch id
        start_col_name: str
            Name of the column containing the start vertices
        batch_col_name: str
            Name of the column containing the batch ids

        Returns
        -------
        None

        Examples
        --------
        >>> import cudf
        >>> from cugraph.experimental.gnn import BulkSampler
        >>> from cugraph.experimental.datasets import karate
        >>> import tempfile
        >>> df = cudf.DataFrame({
        ...     "start_vid": [0, 4, 2, 3, 9, 11],
        ...     "start_batch": cudf.Series(
        ...         [0, 0, 0, 1, 1, 1], dtype="int32")})
        >>> output_tempdir = tempfile.TemporaryDirectory()
        >>> bulk_sampler = BulkSampler(
        ...     batch_size=3,
        ...     output_path=output_tempdir.name,
        ...     graph=karate.get_graph(fetch=True))
        >>> bulk_sampler.add_batches(
        ...     df,
        ...     start_col_name="start_vid",
        ...     batch_col_name="start_batch")
        """
        df = df[[start_col_name, batch_col_name]].rename(
            columns={
                start_col_name: self.start_col_name,
                batch_col_name: self.batch_col_name,
            }
        )

        if self.__batches is None:
            self.__batches = df
        else:
            if isinstance(df, type(self.__batches)):
                if isinstance(df, dask_cudf.DataFrame):
                    concat_fn = dask_cudf.concat
                else:
                    concat_fn = cudf.concat
                self.__batches = concat_fn([self.__batches, df])
            else:
                raise TypeError(
                    "Provided batches must match the dataframe"
                    " type of previous batches!"
                )

        if self.size >= self.seeds_per_call:
            self.__logger.info(
                f"Number of input seeds ({self.size})"
                f" is >= seeds per call ({self.seeds_per_call})."
                " Calling flush() to compute and write minibatches."
            )
            self.flush()

    def flush(self) -> None:
        """
        Computes all uncomputed batches
        """
        if self.size == 0:
            return

        start_time_calc_batches = time.perf_counter()
        if isinstance(self.__batches, dask_cudf.DataFrame):
            self.__batches = self.__batches.persist()

        min_batch_id = self.__batches[self.batch_col_name].min()
        if isinstance(self.__batches, dask_cudf.DataFrame):
            min_batch_id = min_batch_id.persist()
        else:
            min_batch_id = int(min_batch_id)

        partition_size = self.batches_per_partition * self.batch_size
        partitions_per_call = (
            self.seeds_per_call + partition_size - 1
        ) // partition_size
        npartitions = partitions_per_call

        max_batch_id = min_batch_id + npartitions * self.batches_per_partition - 1
        if isinstance(self.__batches, dask_cudf.DataFrame):
            max_batch_id = max_batch_id.persist()

        batch_id_filter = self.__batches[self.batch_col_name] <= max_batch_id
        if isinstance(batch_id_filter, dask_cudf.Series):
            batch_id_filter = batch_id_filter.persist()

        end_time_calc_batches = time.perf_counter()
        self.__logger.info(
            f"Calculated batches to sample; min = {min_batch_id}"
            f" and max = {max_batch_id};"
            f" took {end_time_calc_batches - start_time_calc_batches:.4f} s"
        )

        if isinstance(self.__graph._plc_graph, pylibcugraph.graphs.SGGraph):
            sample_fn = cugraph.uniform_neighbor_sample
        else:
            sample_fn = cugraph.dask.uniform_neighbor_sample
            self.__sample_call_args.update(
                {
                    "_multiple_clients": True,
                    "keep_batches_together": True,
                    "min_batch_id": min_batch_id,
                    "max_batch_id": max_batch_id,
                }
            )

        start_time_sample_call = time.perf_counter()

        # Call uniform neighbor sample
        samples, offsets = sample_fn(
            self.__graph,
            **self.__sample_call_args,
            start_list=self.__batches[[self.start_col_name, self.batch_col_name]][
                batch_id_filter
            ],
            with_batch_ids=True,
            with_edge_properties=True,
            return_offsets=True,
        )

        end_time_sample_call = time.perf_counter()
        sample_runtime = end_time_sample_call - start_time_sample_call

        self.__logger.info(
            f"Called uniform neighbor sample, took {sample_runtime:.4f} s"
        )

        # Filter batches to remove those already processed
        self.__batches = self.__batches[~batch_id_filter]
        del batch_id_filter
        if isinstance(self.__batches, dask_cudf.DataFrame):
            self.__batches = self.__batches.persist()

        start_time_write = time.perf_counter()

        # Write batches to parquet
        self.__write(samples, offsets)
        if isinstance(self.__batches, dask_cudf.DataFrame):
            wait(
                [f.release() for f in futures_of(samples)]
                + [f.release() for f in futures_of(offsets)]
            )

        del samples
        del offsets

        end_time_write = time.perf_counter()
        write_runtime = end_time_write - start_time_write
        self.__logger.info(f"Wrote samples to parquet, took {write_runtime} seconds")

        current_size = self.size
        if current_size > 0:
            self.__logger.info(
                f"There are still {current_size} samples remaining, "
                "calling flush() again..."
            )
            self.flush()

    def __write(
        self,
        samples: Union[cudf.DataFrame, dask_cudf.DataFrame],
        offsets: Union[cudf.DataFrame, dask_cudf.DataFrame],
    ) -> None:
        os.makedirs(self.__output_path, exist_ok=True)
        write_samples(
            samples, offsets, self.__batches_per_partition, self.__output_path
        )
