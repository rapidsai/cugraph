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
            self.df = cudf.DataFrame()

            tmp = df[src_col_names].groupby(src_col_names).count().reset_index().\
                  rename(dict(zip(src_col_names, self.col_names)))

            if dst_col_names is not None:
                tmp_dst = df[dst_col_names].groupby(dst_col_names).count().reset_index()
                for newname, oldname in zip(self.col_names, dst_col_names):
                    self.df[newname] = tmp[newname].append(tmp_dst[oldname])

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

        def add_vertex_id(self, df, id_column_name, col_names):
            if col_names is None:
                return df.merge(self.df, on=self.col_names, how='left').rename({'id': id_column_name})
            else:
                return df.merge(self.df, left_on=col_names, right_on=self.col_names, how='left').rename({'id': id_column_name}).drop(self.col_names)


        def from_vertex_id(self, df, internal_column_name, external_column_names):
            tmp_df = df.merge(self.df, left_on=internal_column_name, right_on='id', how='left')
            if internal_column_name != "id":
                tmp_df = tmp_df.drop(['id'])
            if external_column_names is None:
                return tmp_df
            else:
                return tmp_df.rename(dict(zip(self.col_names, external_column_names)))

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
            return ddf.merge(self.ddf, left_on=col_names, right_on=self.col_names, how='left')['global_id']

        def add_vertex_id(self, ddf, id_column_name, col_names):
            if col_names is None:
                return ddf.merge(self.ddf, on=self.col_names, how='left').reset_index()
            else:
                return ddf.merge(self.ddf, left_on=col_names, right_on=self.col_names).\
                    map_partitions(lambda df: df.drop(self.col_names).rename({'global_id' : id_column_name}))


        def from_vertex_id(self, df, internal_column_name, external_column_names):
            tmp_df = df.merge(self.ddf, left_on=internal_column_name, right_on='global_id', how='left')\
                       .map_partitions(lambda df: df.drop('global_id'))

            if external_column_names is None:
                return tmp_df
            else:
                return tmp_df.rename(dict(zip(self.col_names, external_column_names)))

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
            The vertex identifiers.  Note that to_vertex_id does not guarantee
            order or partitioning (in the case of dask_cudf) of vertex ids.
            If order matters use add_vertex_id

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

    def add_vertex_id(self, df, id_column_name='id', col_names=None):
        """
        Given a collection of external vertex ids, return the internal vertex ids
        combined with the input data.

        If a series-type input is provided then the series will be in a column named '0'.
        Otherwise the input column names in the data frame will be preserved.

        Parameters
        ----------
        df: cudf.DataFrame, cudf.Series, dask_cudf.DataFrame, dask_cudf.Series
            Contains a list of external vertex identifiers that will be
            converted into internal vertex identifiers

        id_column_name: (optional) string
            The name to be applied to the column containing the id
            (defaults to 'id')

        col_names: (optional) list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier
        
        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            A data frame containing the input data (data frame or series)
            with an additional column containing the internal vertex id.
            Note that there is no guarantee of the order or partitioning
            of elements in the returned data frame.

        """
        tmp_df = None
        tmp_colnames = None
        if type(df) is cudf.Series:
            tmp_df = df.to_frame('0')
            tmp_col_names = ['0']
        elif type(df) is dask_cudf.Series:
            tmp_df = df.to_frame('0')
            tmp_col_names = ['0']
        else:
            tmp_df = df
            tmp_col_names = col_names
            
        return self.implementation.add_vertex_id(tmp_df, id_column_name, tmp_col_names)

    def from_vertex_id(self, df, internal_column_name=None, external_column_names=None):
        """
        Given a collection of internal vertex ids, return a data frame of
        the external vertex ids

        Parameters
        ----------
        df: cudf.DataFrame, cudf.Series, dask_cudf.DataFrame, dask_cudf.Series
            A list of internal vertex identifiers that will be
            converted into external vertex identifiers.  If df is a series type
            object it will be converted to a dataframe where the series is
            in a column labeled 'id'.  If df is a dataframe type object
            then internal_column_name should identify which column corresponds
            the the internal vertex id that should be converted

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original data frame columns exist unmodified.  Columns
            are added to the data frame to identify the external vertex identifiers.
            If external_columns is specified, these names are used as the names
            of the output columns.  If external_columns is not specifed the
            columns are labeled '0', ... 'n-1' based on the number of columns
            identifying the external vertex identifiers.
        """
        tmp_df = None
        tmp_colnames = None
        if type(df) is cudf.Series:
            tmp_df = df.to_frame('id')
            internal_column_name = 'id'
        elif type(df) is dask_cudf.Series:
            tmp_df = df.to_frame('id')
            internal_column_name = 'id'
        else:
            tmp_df = df
            
        return self.implementation.from_vertex_id(tmp_df, internal_column_name, external_column_names)

    def column_names(self):
        return self.implementation.col_names
