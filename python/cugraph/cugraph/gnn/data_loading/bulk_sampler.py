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
import cugraph

class EXPERIMENTAL__BulkSampler:
    start_col_name = '_START_'
    batch_col_name = '_BATCH_'
    def __init__(self, output_path:str, graph, saturation_level:int=200_000, rank:int=0, **kwargs):
        """
        Constructs a new BulkSampler

        Parameters
        ----------
        output_path: str
            The directory where results will be stored.
        graph: cugraph.Graph
            The cugraph graph to operate upon.
        saturation_level: int (optional, default=200,000)
            The number of samples that can be made within a single call.
        rank: int (optional, default=0)
            The rank of this sampler.  Used to isolate this sampler from
            others that may be running on other nodes.
        kwargs: kwargs
            Keyword arguments to be passed to the sampler (i.e. fanout).
        """
        self.__output_path = output_path
        self.__graph = graph
        self.__saturation_level = saturation_level
        self.__rank = rank
        self.__batches = None
        self.__sample_call_args = kwargs
    
    @property
    def rank(self) -> int:
        return self.__rank
    
    @property
    def saturation_level(self) -> int:
        return self.__saturation_level
    
    @property
    def size(self) -> int:
        if self.__batches is None:
            return 0
        else:
            return len(self.__batches)

    def add_batches(self, df: Union[cudf.DataFrame, dask_cudf.DataFrame], start_col_name:str, batch_col_name:str) -> None:
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
        """
        df = df.rename(columns={start_col_name: self.start_col_name, batch_col_name: self.batch_col_name})

        if self.__batches is None:
            self.__batches = df
        else:
            if type(df) == type(self.__batches):
                self.__batches = self.__batches.append(df)
            else:
                raise TypeError('Provided batches must match the dataframe type of previous batches!')
        
        if self.size >= self.saturation_level:
            self.flush()

    def flush(self) -> None:
        """
        Computes all uncomputed batches
        """
        end = min(self.__saturation_level, len(self.__batches))

        sample_fn = cugraph.dask.uniform_neighbor_sample if isinstance(self.__batches, dask_cudf.DataFrame) else cugraph.uniform_neighbor_sample

        # TODO semaphore check to prevent concurrent calls to uniform_neighbor_sample
        samples = sample_fn(
            self.__graph,
            **self.__sample_call_args,
            start_list=self.__batches[self.start_col_name][:end],
            batch_id_list=self.__batches[self.batch_col_name][:end],
            with_edge_properties=True,
        )

        if len(self.__batches) > end:
            self.__batches = self.__batches[end:]
        else:
            self.__batches = None
        
        self.__write(samples)

    def __write(self, samples:Union[cudf.DataFrame, dask_cudf.DataFrame]) -> None:
        # Ensure each rank writes to its own partition so there is no conflict
        outer_partition = f'rank={self.__rank}'
        if isinstance(samples, dask_cudf.DataFrame):
            samples.to_parquet(
                os.path.join(self.__output_path, outer_partition),
                partition_on=['batch_id', 'hop_id']
            )
        else:
            samples.to_parquet(
                os.path.join(self.__output_path, outer_partition),
                partition_cols=['batch_id', 'hop_id']
            )