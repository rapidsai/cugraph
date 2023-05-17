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

import cupy
import cudf
import dask_cudf
import cugraph.dask as dask_cugraph

import cugraph
import pylibcugraph

from cugraph.gnn.data_loading.bulk_sampler_io import write_samples


class EXPERIMENTAL__BulkSampler:
    start_col_name = "_START_"
    batch_col_name = "_BATCH_"

    def __init__(
        self,
        batch_size: int,
        output_path: str,
        graph,
        seeds_per_call: int = 200_000,
        batches_per_partition=100,
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
        kwargs: kwargs
            Keyword arguments to be passed to the sampler (i.e. fanout).
        """

        max_batches_per_partition = seeds_per_call // batch_size
        if batches_per_partition > max_batches_per_partition:
            import warnings

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
        df = df.rename(
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
            self.flush()

    def flush(self) -> None:
        """
        Computes all uncomputed batches
        """
        if self.size == 0:
            return
        self.__batches.reset_index(drop=True)

        min_batch_id = self.__batches[self.batch_col_name].min()
        if isinstance(self.__batches, dask_cudf.DataFrame):
            min_batch_id = min_batch_id.compute()
        min_batch_id = int(min_batch_id)

        partition_size = self.batches_per_partition * self.batch_size
        partitions_per_call = (
            self.seeds_per_call + partition_size - 1
        ) // partition_size
        npartitions = partitions_per_call

        max_batch_id = min_batch_id + npartitions * self.batches_per_partition - 1
        batch_id_filter = self.__batches[self.batch_col_name] <= max_batch_id

        if isinstance(self.__graph._plc_graph, pylibcugraph.graphs.SGGraph):
            sample_fn = cugraph.uniform_neighbor_sample
        else:
            sample_fn = cugraph.dask.uniform_neighbor_sample
            self.__sample_call_args.update(
                {
                    "_multiple_clients": True,
                    "label_to_output_comm_rank": self.__get_label_to_output_comm_rank(
                        min_batch_id, max_batch_id
                    ),
                    "label_list": cupy.arange(
                        min_batch_id, max_batch_id + 1, dtype="int32"
                    ),
                }
            )

        samples, offsets = sample_fn(
            self.__graph,
            **self.__sample_call_args,
            start_list=self.__batches[self.start_col_name][batch_id_filter],
            batch_id_list=self.__batches[self.batch_col_name][batch_id_filter],
            with_edge_properties=True,
            return_offsets=True,
        )
        print('samples!')
        print(samples)

        self.__batches = self.__batches[~batch_id_filter]
        self.__write(samples, offsets)

        if self.size > 0:
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

    def __get_label_to_output_comm_rank(self, min_batch_id, max_batch_id):
        num_workers = dask_cugraph.get_n_workers()
        num_batches = max_batch_id - min_batch_id + 1
        z = cupy.zeros(num_batches, dtype="int32")
        s = cupy.array_split(cupy.arange(num_batches), num_workers)
        for i, t in enumerate(s):
            z[t] = i

        return cudf.Series(z)
