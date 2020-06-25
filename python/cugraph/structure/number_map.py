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

class NumberMap:
    class SingleGPU:
        def __init__(self, df, col_names):
            self.col_names = NumberMap.compute_vals(col_names)
            tmp = df[col_names].groupby(col_names).count().reset_index()
            self.df = tmp.rename(dict(zip(col_names, self.col_names)))
            self.numbered = False

        def append(self, df, col_names):
            if self.numbered:
                raise Exception("Can't append data once the compute function has been called")

            newdf = type(self.df)()
            tmp = df[col_names].groupby(col_names).count().reset_index()
            for newname, oldname in zip(self.col_names, col_names):
                newdf[newname] = self.df[newname].append(tmp[oldname])

            self.df = newdf

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
        def __init__(self, ddf, col_names):
            # TODO: Fill in implementation
            pass

        def append(self, ddf, col_names):
            # TODO: Fill in implementation
            pass

        def compute(self):
            # TODO: Fill in implementation
            pass

        def to_vertex_id(self, df, col_names):
            # TODO: Fill in implementation
            pass

        def from_vertex_id(self, series):
            # TODO: Fill in implementation
            pass

            
    def __init__(self):
        self.implementation = None

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
            self.implementation = NumberMap.SingleGPU(df, src_col_names)
        elif type(df) is dask_cudf.DataFrame:
            self.implementation = NumberMap.MultiGPU(df, src_col_names)
        else:
            raise Exception('df must be cudf.DataFrame or dask_cudf.DataFrame')

        if dst_col_names is not None:
            self.implementation.append(df, dst_col_names)

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
            df = cudf.DataFrame()
            df['s'] = src_series
            self.implementation = NumberMap.SingleGPU(df, ['s'])
        elif type(src_series) is dask_cudf.Series:
            df = dask_cudf.DataFrame()
            df['s'] = src_series
            self.implementation = NumberMap.MultiGPU(df, ['s'])
        else:
            raise Exception('src_series must be cudf.Series or dask_cudf.Series')

        if dst_series is not None:
            df['d'] = dst_series
            self.implementation.append(df, ['d'])

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
