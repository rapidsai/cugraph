# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cugraph
import cudf
import dask_cudf
import numpy as np
import bisect

class NumberMap:
    class SingleGPU:
        def __init__(self, df, src_col_names, dst_col_names):
            self.col_names = NumberMap.compute_vals(src_col_names)
            tmp = df[src_col_names].groupby(src_col_names).count().reset_index().\
                  rename(dict(zip(src_col_names, self.col_names)))

            if dst_col_names is not None:
                tmp_dst = df[dst_col_names].groupby(dst_col_names).count().reset_index()
                for newname, oldname in zip(self.col_names, dst_col_names):
                    tmp[newname] = tmp[newname].append(tmp_dst[oldname])

            self.df = tmp
            self.numbered = False

        def compute(self):
            if not self.numbered:
                tmp = self.df.groupby(self.col_names).count().reset_index()
                tmp['id'] = tmp.index.astype(np.int32)
                self.df = tmp
                self.numbered = True

        def to_vertex_id(self, df, col_names):
            tmp_df = df[col_names].rename(dict(zip(col_names, self.col_names)))
            tmp_df['index'] = tmp_df.index
            return tmp_df.merge(self.df, on=self.col_names, how='left').sort_values('index').drop(['index']).reset_index()['id']

        def from_vertex_id(self, series):
            tmp_df = cudf.DataFrame()
            tmp_df['id'] = series
            tmp_df['index'] = tmp_df.index
            return tmp_df.merge(self.df, on=['id'], how='left').sort_values('index').drop(['index', 'id']).reset_index()[self.col_names]

    class MultiGPU:
        def extract_vertices(df, src_col_names, dst_col_names, internal_col_names):
            s = df[src_col_names].groupby(src_col_names).count().reset_index().\
                rename(dict(zip(src_col_names, internal_col_names)))
            d = None

            if dst_col_names is not None:
                d = df[dst_col_names].groupby(dst_col_names).count().reset_index().\
                    rename(dict(zip(dst_col_names, internal_col_names)))

            reply = cudf.DataFrame()

            for i in internal_col_names:
                if d is None:
                    reply[i] = s[i]
                else:
                    reply[i] = s[i].append(d[i])

            return reply

        def __init__(self, ddf, src_col_names, dst_col_names):
            self.col_names = NumberMap.compute_vals(src_col_names)
            self.val_types = NumberMap.compute_vals_types(ddf, src_col_names)
            self.ddf = ddf.map_partitions(NumberMap.MultiGPU.extract_vertices, src_col_names, dst_col_names, self.col_names, meta=self.val_types)
            self.numbered = False

        # Function to compute partitions based on known divisions of the
        # hash value
        def compute_partition(df, divisions):
            sample = df.index[0]
            partition_id = bisect.bisect_right(divisions, sample) - 1
            return df.assign(partition = partition_id)

        def assign_internal_identifiers_kernel(local_id, partition, global_id, base_addresses):
            for i in range(len(local_id)):
                global_id[i] = local_id[i] + base_addresses[partition[i]]

        def compute(self):
            if not self.numbered:
                val_types  = self.val_types
                val_types['hash'] = np.int32

                vertices = self.ddf.map_partitions(lambda df: df.assign(hash = df.hash_columns(self.col_names)), meta=val_types)

                # Redistribute the ddf based on the hash values
                rehashed = vertices.set_index('hash', drop=False)

                # Compute the local partition id (obsolete once
                #   https://github.com/dask/dask/issues/3707 is completed)
                val_types['partition'] = np.int32
                rehashed_with_partition_id = rehashed.map_partitions(NumberMap.MultiGPU.compute_partition, rehashed.divisions, meta=val_types)

                numbering_map = rehashed_with_partition_id.map_partitions(lambda df: df.groupby(self.col_names).min().reset_index())

                #
                #  Compute base address for each partition
                #
                counts = numbering_map.map_partitions(lambda df: df.groupby('partition').count()).compute()['hash']
                base_addresses = cudf.Series(np.zeros(len(counts) + 1, np.int64))
                for i in range(len(counts)):
                    base_addresses[i+1] = base_addresses[i] + counts[i]

                #
                #  Update each partition with the base address
                #
                val_types['global_id'] = np.int32
                del val_types['hash']
                del val_types['partition']
                numbering_map = numbering_map.map_partitions(lambda df:
                                                             df.assign(local_id = df.index.astype(np.int64)).
                                                             apply_rows(NumberMap.MultiGPU.assign_internal_identifiers_kernel,
                                                                        incols=['local_id', 'partition'],
                                                                        outcols={'global_id': np.int64},
                                                                        kwargs={'base_addresses': base_addresses}).drop(['local_id', 'hash', 'partition']),
                                                             meta=val_types)

                self.ddf = numbering_map
                self.numbered = True

        def to_vertex_id(self, ddf, col_names):
            tmp_ddf = ddf
            #tmp_ddf = ddf.map_partitions(lambda df: df.assign(idx = df.index.astype(np.int32)))
            #print('to_vertex_id, col_names = ', col_names)
            #print('self.ddf = ', self.ddf.compute())
            #print('tmp_df = ', tmp_ddf)
            tmp_ddf['idx'] = tmp_ddf.index
            print('tmp_ddf = ', tmp_ddf.compute())
            #return tmp_ddf.merge(self.ddf, left_on=col_names, right_on=self.col_names, how='left').sort_values('index')['global_id']
            #xxx = tmp_ddf.merge(self.ddf, left_on=col_names, right_on=self.col_names, how='left').sort_values('idx')['global_id']
            xxx = tmp_ddf.merge(self.ddf, left_on=col_names, right_on=self.col_names, how='left').sort_values('idx')
            print('xxx = ', xxx.compute())
            return xxx.reset_index()['global_id']

        def from_vertex_id(self, series):
            tmp_ddf = series.to_frame('global_id')
            tmp_ddf['idx'] = tmp_ddf.index

            print('from_vertex_id, tmp_ddf = ', tmp_ddf.map_partitions(lambda df: df.head()).compute())
            print('series = ', series.compute())
            print('self.ddf = ', self.ddf.compute())
            #return tmp_ddf.merge(self.ddf, on='global_id', how='left').sort_values('idx')[self.col_names]
            xxx = tmp_ddf.merge(self.ddf, on='global_id', how='left').map_partitions(lambda df: df.sort_values('idx'))
            #.sort_values('idx')
            #print('xxx = ', xxx.head())
            print('xxx = ', xxx.map_partitions(lambda df: df.head()).compute())

            return xxx[self.col_names]


            
    def __init__(self):
        self.implementation = None

    def compute_vals_types(df, column_names):
        """
        Helper function to compute internal column names and types
        """
        return {str(i): df[column_names[i]].dtype  for i in range(len(column_names))}
                                          
    def compute_vals(column_names):
        """
        Helper function to compute internal column names based on external column names
        """
        return [str(i) for i in range(len(column_names))]

    def from_dataframe(self, df, src_col_names, dst_col_names=None):
        """
        Populate the numbering map with vertices from the specified
        columns of the provided data frame.

        Parameters
        ----------
        df : cudf.DataFrame or dask_cudf.DataFrame
            Contains a list of external vertex identifiers that will be
            numbered by the NumberMap class.
        src_col_names: list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier for source vertices
        dst_col_names: list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier for destination vertices
        """
        if self.implementation is not None:
            raise Exception('NumberMap is already populated')

        if dst_col_names is not None and len(src_col_names) != len(dst_col_names):
            raise Exception('src_col_names must have same length as dst_col_names')

        if type(df) is cudf.DataFrame:
            self.implementation = NumberMap.SingleGPU(df, src_col_names, dst_col_names)
        elif type(df) is dask_cudf.DataFrame:
            self.implementation = NumberMap.MultiGPU(df, src_col_names, dst_col_names)
        else:
            raise Exception('df must be cudf.DataFrame or dask_cudf.DataFrame')

        self.implementation.compute()

    def from_series(self, src_series, dst_series=None):
        """
        Populate the numbering map with vertices from the specified
        pair of series objects, one for the source and one for
        the destination

        Parameters
        ----------
        src_series: cudf.Series or dask_cudf.Series
            Contains a list of external vertex identifiers that will be
            numbered by the NumberMap class.
        dst_series: cudf.Series or dask_cudf.Series
            Contains a list of external vertex identifiers that will be
            numbered by the NumberMap class.
        """
        if self.implementation is not None:
            raise Exception('NumberMap is already populated')

        if dst_series is not None and type(src_series) != type(dst_series):
            raise Exception('src_series and dst_series must have same type')

        if type(src_series) is cudf.Series:
            dst_series_list = None
            df = cudf.DataFrame()
            df['s'] = src_series
            if dst_series is not None:
                df['d'] = dst_series
                dst_series_list = ['d']
            self.implementation = NumberMap.SingleGPU(df, ['s'], dst_series_list)
        elif type(src_series) is dask_cudf.Series:
            dst_series_list = None
            df = dask_cudf.DataFrame()
            df['s'] = src_series
            if dst_series is not None:
                df['d'] = dst_series
                dst_series_list = ['d']
            self.implementation = NumberMap.MultiGPU(df, ['s'], dst_series_list)
        else:
            raise Exception('src_series must be cudf.Series or dask_cudf.Series')

        self.implementation.compute()

    def to_vertex_id(self, df, col_names=None):
        """
        Given a collection of external vertex ids, return the internal vertex ids

        Parameters
        ----------
        df: cudf.DataFrame, cudf.Series, dask_cudf.DataFrame, dask_cudf.Series
            Contains a list of external vertex identifiers that will be
            converted into internal vertex identifiers

        col_names: (optional) list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier
        
        Returns
        ---------
        vertex_ids : cudf.Series or dask_cudf.Series
            The vertex identifiers

        """
        tmp_df = None
        tmp_colnames = None
        if type(df) is cudf.Series:
            tmp_df = cudf.DataFrame()
            tmp_df['0'] = df
            tmp_col_names = ['0']
        elif type(df) is dask_cudf.Series:
            tmp_df = dask_cudf.DataFrame()
            tmp_df['0'] = df
            tmp_col_names = ['0']
        else:
            tmp_df = df
            tmp_col_names = col_names
            
        return self.implementation.to_vertex_id(tmp_df, tmp_col_names)

    def from_vertex_id(self, series):
        """
        Given a collection of internal vertex ids, return a data frame of
        the external vertex ids

        Parameters
        ----------
        series : cudf.Series or dask_cudf.Series
            A list of internal vertex identifiers that will be
            converted into external vertex identifiers

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The external vertex identifiers.  Columns are labeled
            '0', ... 'n-1' based on the number of columns identifying
            the external vertex identifiers.
        """
        return self.implementation.from_vertex_id(series)
