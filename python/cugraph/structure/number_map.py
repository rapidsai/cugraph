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

class NumberMap:
    class SingleGPU:
        def __init__(self, df, col_names):
            # TODO: Fill in implementation
            pass

        def append(self, df, col_names):
            # TODO: Fill in implementation
            pass

        def compute(self):
            # TODO: Fill in implementation
            pass

        def to_vertex_id(df, col_names):
            # TODO: Fill in implementation
            pass

        def from_vertex_id(series):
            # TODO: Fill in implementation
            pass

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

        def to_vertex_id():
            # TODO: Fill in implementation
            pass

        def from_vertex_id():
            # TODO: Fill in implementation
            pass

            
    def __init__(self):
        self.implementation = None

    def compute_vals(column_names):
        """
        Helper function to compute internal column names based on external column names
        """
        return [str(i) for i in range(len(column_names))]

    def from_dataframe(self, df, col_names):
        """
        Populate the numbering map with vertices from the specified
        columns of the provided data frame.

        Parameters
        ----------
        df : cudf.DataFrame or dask_cudf.DataFrame
            Contains a list of external vertex identifiers that will be
            numbered by the NumberMap class.
        col_names: list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier
        """
        if self.implementation is not None:
            raise Exception('NumberMap is already populated')

        if type(df) is cudf.DataFrame:
            self.implementation = NumberMap.SingleGPU(df, col_names)
        elif type(df) is dask_cudf.DataFrame:
            self.implementation = NumberMap.MultiGPU(df, col_names)
        else:
            raise Exception('df must be cudf.DataFrame or dask_cudf.DataFrame')

        self.implementation.compute()

    def from_cudf_dataframe(self, df, src_col_names, dst_col_names):
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

        if len(src_col_names) != len(dst_col_names):
            raise Exception('src_col_names must have same length as dst_col_names')

        if type(df) is cudf.DataFrame:
            self.implementation = NumberMap.SingleGPU(df, src_col_names)
        elif type(df) is dask_cudf.DataFrame:
            self.implementation = NumberMap.MultiGPU(df, src_col_names)
        else:
            raise Exception('df must be cudf.DataFrame or dask_cudf.DataFrame')

        self.implementation.append(df, dst_col_names)
        self.implementation.compute()

    def from_cudf_series(self, series):
        """
        Populate the numbering map with vertices from the specified
        series

        Parameters
        ----------
        series : cudf.Series or dask_cudf.Series
            Contains a list of external vertex identifiers that will be
            numbered by the NumberMap class.
        """
        if self.implementation is not None:
            raise Exception('NumberMap is already populated')

        if type(series) is cudf.Series:
            df = cudf.DataFrame()
            df['0'] = series
            self.implementation = NumberMap.SingleGPU(series)
        elif type(series) is dask_cudf.Series:
            df = dask_cudf.DataFrame()
            df['0'] = series
            self.implementation = NumberMap.MultiGPU(series)
        else:
            raise Exception('series must be cudf.Series or dask_cudf.Series')

        self.implementation.compute()

    def from_cudf_series(self, src_series, dst_series):
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

        if type(src_series) != type(dst_series):
            raise Exception('src_series and dst_series must have same type')

        if type(src_series) is cudf.Series:
            df = cudf.DataFrame()
            df['s'] = src_series
            df['d'] = dst_series
            self.implementation = NumberMap.SingleGPU(df, ['s'])
        elif type(src_series) is dask_cudf.Series:
            df = dask_cudf.DataFrame()
            df['s'] = src_series
            df['d'] = dst_series
            self.implementation = NumberMap.MultiGPU(df, ['s'])
        else:
            raise Exception('src_series must be cudf.Series or dask_cudf.Series')

        self.implementation.append(df, ['d'])
        self.implementation.compute()

    def to_vertex_id(self, df, col_names):
        """
        Given a collection of external vertex ids, return the internal vertex ids

        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            Contains a list of external vertex identifiers that will be
            converted into internal vertex identifiers

        col_names: list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier
        
        Returns
        ---------
        vertex_ids : cudf.Series or dask_cudf.Series
            The vertex identifiers

        """
        return self.implementation.to_vertex_id(df, col_names)

    def to_vertex_id(self, series):
        """
        Given a collection of external vertex ids, return the internal vertex ids

        Parameters
        ----------
        series: cudf.Series or dask_cudf.Series
            Contains a list of external vertex identifiers that will be
            converted into internal vertex identifiers

        Returns
        ---------
        vertex_ids : cudf.Series or dask_cudf.Series
            The vertex identifiers

        """
        if type(series) is cudf.Series:
            df = cudf.DataFrame()
            df['0'] = series
        elif type(series) is dask_cudf.Series:
            df = dask_cudf.DataFrame()
            df['0'] = src_series
        else:
            raise Exception('series must be cudf.Series or dask_cudf.Series')

        return self.implementation.to_vertex_id(df, ['0'])

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
