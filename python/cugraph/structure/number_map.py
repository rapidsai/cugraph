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

import cudf
import dask_cudf
import numpy as np
import bisect


class NumberMap:
    """
    Class used to translate external vertex ids to internal vertex ids
    in the cuGraph framework.

    Internal vertex ids are assigned by hashing the external vertex ids
    into a structure to eliminate duplicates, and the resulting list
    of unique vertices are assigned integers from [0, V) where V is
    the number of unique vertices.

    In Single GPU mode, internal vertex ids are constructed using
    cudf functions, with a cudf.DataFrame containing the mapping
    from external vertex identifiers and internal vertex identifiers
    allowing for mapping vertex identifiers in either direction.  In
    this mode, the order of the output from the mapping functions is
    non-deterministic.  cudf makes no guarantees about order.  If
    matching the input order is required set the preserve_order
    to True.

    In Multi GPU mode, internal vertex ids are constucted using
    dask_cudf functions, with a dask_cudf.DataFrame containing
    the mapping from external vertex identifiers and internal
    vertex identifiers allowing for mapping vertex identifiers
    in either direction.  In this mode, the partitioning of
    the number_map and the output from any of the mapping functions
    are non-deterministic.  dask_cudf makes no guarantees about the
    partitioning or order of the output.  As of this release,
    there is no mechanism for controlling that, this will be
    addressed at some point.
    """

    class SingleGPU:
        def __init__(self, df, src_col_names, dst_col_names, id_type,
                     store_transposed):
            self.col_names = NumberMap.compute_vals(src_col_names)
            self.df = cudf.DataFrame()
            self.id_type = id_type
            self.store_transposed = store_transposed

            source_count = 0
            dest_count = 0

            if store_transposed:
                dest_count = 1
            else:
                source_count = 1

            tmp = (
                df[src_col_names]
                .assign(count=source_count)
                .groupby(src_col_names)
                .sum()
                .reset_index()
                .rename(
                    columns=dict(zip(src_col_names, self.col_names)),
                    copy=False,
                )
            )

            if dst_col_names is not None:
                tmp_dst = (
                    df[dst_col_names]
                    .assign(count=dest_count)
                    .groupby(dst_col_names)
                    .sum()
                    .reset_index()
                )
                for newname, oldname in zip(self.col_names, dst_col_names):
                    self.df[newname] = tmp[newname].append(tmp_dst[oldname])
                self.df['count'] = tmp['count'].append(tmp_dst['count'])
            else:
                for newname, oldname in zip(self.col_names, dst_col_names):
                    self.df[newname] = tmp[newname]
                self.df['count'] = tmp['count']

            self.numbered = False

        def compute(self):
            if not self.numbered:
                tmp = self.df.groupby(self.col_names).sum().sort_values(
                    'count', ascending=False
                ).reset_index().drop(columns='count')

                tmp["id"] = tmp.index.astype(self.id_type)
                self.df = tmp
                self.numbered = True

        def to_internal_vertex_id(self, df, col_names):
            tmp_df = df[col_names].rename(
                columns=dict(zip(col_names, self.col_names)), copy=False
            )
            index_name = NumberMap.generate_unused_column_name(df.columns)
            tmp_df[index_name] = tmp_df.index
            return (
                self.df.merge(tmp_df, on=self.col_names, how="right")
                .sort_values(index_name)
                .drop(columns=[index_name])
                .reset_index()["id"]
            )

        def add_internal_vertex_id(self, df, id_column_name, col_names,
                                   drop, preserve_order):
            ret = None

            if preserve_order:
                index_name = NumberMap.generate_unused_column_name(df.columns)
                tmp_df = df
                tmp_df[index_name] = tmp_df.index
            else:
                tmp_df = df

            if "id" in df.columns:
                id_name = NumberMap.generate_unused_column_name(tmp_df.columns)
                merge_df = self.df.rename(columns={"id": id_name}, copy=False)
            else:
                id_name = "id"
                merge_df = self.df

            if col_names is None:
                ret = merge_df.merge(tmp_df, on=self.col_names, how="right")
            elif col_names == self.col_names:
                ret = merge_df.merge(tmp_df, on=self.col_names, how="right")
            else:
                ret = (
                    merge_df.merge(
                        tmp_df,
                        right_on=col_names,
                        left_on=self.col_names,
                        how="right",
                    )
                    .drop(columns=self.col_names)
                )

            if drop:
                ret = ret.drop(columns=col_names)

            ret = ret.rename(
                columns={id_name: id_column_name}, copy=False
            )

            if preserve_order:
                ret = ret.sort_values(index_name).reset_index(drop=True)

            return ret

        def from_internal_vertex_id(
            self, df, internal_column_name, external_column_names
        ):
            tmp_df = self.df.merge(
                df,
                right_on=internal_column_name,
                left_on="id",
                how="right",
            )
            if internal_column_name != "id":
                tmp_df = tmp_df.drop(columns=["id"])
            if external_column_names is None:
                return tmp_df
            else:
                return tmp_df.rename(
                    columns=dict(zip(self.col_names, external_column_names)),
                    copy=False,
                )

    class MultiGPU:
        def extract_vertices(
            df, src_col_names, dst_col_names,
            internal_col_names, store_transposed
        ):
            source_count = 0
            dest_count = 0

            if store_transposed:
                dest_count = 1
            else:
                source_count = 1

            s = (
                df[src_col_names]
                .assign(count=source_count)
                .groupby(src_col_names)
                .sum()
                .reset_index()
                .rename(
                    columns=dict(zip(src_col_names, internal_col_names)),
                    copy=False,
                )
            )
            d = None

            if dst_col_names is not None:
                d = (
                    df[dst_col_names]
                    .assign(count=dest_count)
                    .groupby(dst_col_names)
                    .sum()
                    .reset_index()
                    .rename(
                        columns=dict(zip(dst_col_names, internal_col_names)),
                        copy=False,
                    )
                )

            reply = cudf.DataFrame()

            for i in internal_col_names:
                if d is None:
                    reply[i] = s[i]
                else:
                    reply[i] = s[i].append(d[i])

            reply['count'] = s['count'].append(d['count'])

            return reply

        def __init__(
            self, ddf, src_col_names, dst_col_names, id_type, store_transposed
        ):
            self.col_names = NumberMap.compute_vals(src_col_names)
            self.val_types = NumberMap.compute_vals_types(ddf, src_col_names)
            self.val_types["count"] = np.int32
            self.id_type = id_type
            self.store_transposed = store_transposed
            self.ddf = ddf.map_partitions(
                NumberMap.MultiGPU.extract_vertices,
                src_col_names,
                dst_col_names,
                self.col_names,
                store_transposed,
                meta=self.val_types,
            )
            self.numbered = False

        # Function to compute partitions based on known divisions of the
        # hash value
        def compute_partition(df, divisions):
            sample = df.index[0]
            partition_id = bisect.bisect_right(divisions, sample) - 1
            return df.assign(partition=partition_id)

        def assign_internal_identifiers_kernel(
            local_id, partition, global_id, base_addresses
        ):
            for i in range(len(local_id)):
                global_id[i] = local_id[i] + base_addresses[partition[i]]

        def assign_internal_identifiers(df, base_addresses, id_type):
            df = df.assign(local_id=df.index.astype(np.int64))
            df = df.apply_rows(
                NumberMap.MultiGPU.assign_internal_identifiers_kernel,
                incols=["local_id", "partition"],
                outcols={"global_id": id_type},
                kwargs={"base_addresses": base_addresses},
            )

            return df.drop(columns=["local_id", "hash", "partition"])

        def assign_global_id(self, ddf, base_addresses, val_types):
            val_types["global_id"] = self.id_type
            del val_types["hash"]
            del val_types["partition"]

            ddf = ddf.map_partitions(
                lambda df: NumberMap.MultiGPU.assign_internal_identifiers(
                    df, base_addresses, self.id_type
                ),
                meta=val_types,
            )
            return ddf

        def compute(self):
            if not self.numbered:
                val_types = self.val_types
                val_types["hash"] = np.int32

                vertices = self.ddf.map_partitions(
                    lambda df: df.assign(hash=df.hash_columns(self.col_names)),
                    meta=val_types,
                )

                # Redistribute the ddf based on the hash values
                rehashed = vertices.set_index("hash", drop=False)

                # Compute the local partition id (obsolete once
                #   https://github.com/dask/dask/issues/3707 is completed)
                val_types["partition"] = np.int32

                rehashed_with_partition_id = rehashed.map_partitions(
                    NumberMap.MultiGPU.compute_partition,
                    rehashed.divisions,
                    meta=val_types,
                )

                val_types.pop('count')

                numbering_map = rehashed_with_partition_id.map_partitions(
                    lambda df: df.groupby(
                        self.col_names + ["hash", "partition"]
                    ).sum()
                    .sort_values('count', ascending=False)
                    .reset_index()
                    .drop(columns='count'),
                    meta=val_types
                )

                #
                #  Compute base address for each partition
                #
                counts = numbering_map.map_partitions(
                    lambda df: df.groupby("partition").count()
                ).compute()["hash"].to_pandas()
                base_addresses = np.zeros(len(counts) + 1, self.id_type)

                for i in range(len(counts)):
                    base_addresses[i + 1] = base_addresses[i] + counts[i]

                #
                #  Update each partition with the base address
                #
                numbering_map = self.assign_global_id(
                    numbering_map, cudf.Series(base_addresses), val_types
                )

                self.ddf = numbering_map.persist()
                self.numbered = True

        def to_internal_vertex_id(self, ddf, col_names):
            return self.ddf.merge(
                ddf,
                right_on=col_names,
                left_on=self.col_names,
                how="right",
            )["global_id"]

        def add_internal_vertex_id(self, ddf, id_column_name, col_names, drop,
                                   preserve_order):
            # At the moment, preserve_order cannot be done on
            # multi-GPU
            if preserve_order:
                raise Exception("preserve_order not supported for multi-GPU")

            ret = None
            if col_names is None:
                ret = self.ddf.merge(
                    ddf, on=self.col_names, how="right"
                )
            elif col_names == self.col_names:
                ret = self.ddf.merge(
                    ddf, on=col_names, how="right"
                )
            else:
                ret = self.ddf.merge(
                    ddf, right_on=col_names, left_on=self.col_names
                ).map_partitions(
                    lambda df: df.drop(columns=self.col_names)
                )

            if drop:
                ret = ret.map_partitions(lambda df: df.drop(columns=col_names))

            ret = ret.map_partitions(
                lambda df: df.rename(
                    columns={"global_id": id_column_name}, copy=False
                )
            )

            return ret

        def from_internal_vertex_id(
            self, df, internal_column_name, external_column_names
        ):
            tmp_df = self.ddf.merge(
                df,
                right_on=internal_column_name,
                left_on="global_id",
                how="right"
            ).map_partitions(lambda df: df.drop(columns="global_id"))

            if external_column_names is None:
                return tmp_df
            else:
                return tmp_df.map_partitions(
                    lambda df:
                    df.rename(
                        columns=dict(
                            zip(self.col_names, external_column_names)
                        ),
                        copy=False
                    )
                )

    def __init__(self, id_type=np.int32):
        self.implementation = None
        self.id_type = id_type

    def aggregate_count_and_partition(df):
        d = {}
        d['count'] = df['count'].sum()
        d['partition'] = df['partition'].min()
        return cudf.Series(d, index=['count', 'partition'])

    def compute_vals_types(df, column_names):
        """
        Helper function to compute internal column names and types
        """
        return {
            str(i): df[column_names[i]].dtype for i in range(len(column_names))
        }

    def generate_unused_column_name(column_names):
        """
        Helper function to generate an unused column name
        """
        name = 'x'
        while name in column_names:
            name = name + "x"

        return name

    def compute_vals(column_names):
        """
        Helper function to compute internal column names based on external
        column names
        """
        return [str(i) for i in range(len(column_names))]

    def from_dataframe(
            self, df, src_col_names, dst_col_names=None, store_transposed=False
    ):
        """
        Populate the numbering map with vertices from the specified
        columns of the provided DataFrame.

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
        store_transposed : bool
            Identify how the graph adjacency will be used.
            If True, the graph will be organized by destination.
            If False, the graph will be organized by source

        """
        if self.implementation is not None:
            raise Exception("NumberMap is already populated")

        if dst_col_names is not None and len(src_col_names) != len(
            dst_col_names
        ):
            raise Exception(
                "src_col_names must have same length as dst_col_names"
            )

        if type(df) is cudf.DataFrame:
            self.implementation = NumberMap.SingleGPU(
                df, src_col_names, dst_col_names, self.id_type,
                store_transposed
            )
        elif type(df) is dask_cudf.DataFrame:
            self.implementation = NumberMap.MultiGPU(
                df, src_col_names, dst_col_names, self.id_type,
                store_transposed
            )
        else:
            raise Exception("df must be cudf.DataFrame or dask_cudf.DataFrame")

        self.implementation.compute()

    def from_series(self, src_series, dst_series=None, store_transposed=False):
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
        store_transposed : bool
            Identify how the graph adjacency will be used.
            If True, the graph will be organized by destination.
            If False, the graph will be organized by source
        """
        if self.implementation is not None:
            raise Exception("NumberMap is already populated")

        if dst_series is not None and type(src_series) != type(dst_series):
            raise Exception("src_series and dst_series must have same type")

        if type(src_series) is cudf.Series:
            dst_series_list = None
            df = cudf.DataFrame()
            df["s"] = src_series
            if dst_series is not None:
                df["d"] = dst_series
                dst_series_list = ["d"]
            self.implementation = NumberMap.SingleGPU(
                df, ["s"], dst_series_list, self.id_type, store_transposed
            )
        elif type(src_series) is dask_cudf.Series:
            dst_series_list = None
            df = dask_cudf.DataFrame()
            df["s"] = src_series
            if dst_series is not None:
                df["d"] = dst_series
                dst_series_list = ["d"]
            self.implementation = NumberMap.MultiGPU(
                df, ["s"], dst_series_list, self.id_type, store_transposed
            )
        else:
            raise Exception(
                "src_series must be cudf.Series or " "dask_cudf.Series"
            )

        self.implementation.compute()

    def to_internal_vertex_id(self, df, col_names=None):
        """
        Given a collection of external vertex ids, return the internal
        vertex ids

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
            The vertex identifiers.  Note that to_internal_vertex_id
            does not guarantee order or partitioning (in the case of
            dask_cudf) of vertex ids. If order matters use
            add_internal_vertex_id

        """
        tmp_df = None
        tmp_col_names = None
        if type(df) is cudf.Series:
            tmp_df = cudf.DataFrame()
            tmp_df["0"] = df
            tmp_col_names = ["0"]
        elif type(df) is dask_cudf.Series:
            tmp_df = dask_cudf.DataFrame()
            tmp_df["0"] = df
            tmp_col_names = ["0"]
        else:
            tmp_df = df
            tmp_col_names = col_names

        reply = self.implementation.to_internal_vertex_id(tmp_df,
                                                          tmp_col_names)

        if type(df) in [cudf.DataFrame, dask_cudf.DataFrame]:
            return reply["0"]
        else:
            return reply

    def add_internal_vertex_id(
        self, df, id_column_name="id", col_names=None, drop=False,
        preserve_order=False
    ):
        """
        Given a collection of external vertex ids, return the internal vertex
        ids combined with the input data.

        If a series-type input is provided then the series will be in a column
        named '0'. Otherwise the input column names in the DataFrame will be
        preserved.

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

        drop: (optional) boolean
            If True, drop the column names specified in col_names from
            the returned DataFrame.  Defaults to False.

        preserve_order: (optional) boolean
            If True, do extra sorting work to preserve the order
            of the input DataFrame.  Defaults to False.

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame containing the input data (DataFrame or series)
            with an additional column containing the internal vertex id.
            Note that there is no guarantee of the order or partitioning
            of elements in the returned DataFrame.

        """
        tmp_df = None
        tmp_col_names = None
        can_drop = True
        if type(df) is cudf.Series:
            tmp_df = df.to_frame("0")
            tmp_col_names = ["0"]
            can_drop = False
        elif type(df) is dask_cudf.Series:
            tmp_df = df.to_frame("0")
            tmp_col_names = ["0"]
            can_drop = False
        else:
            tmp_df = df

            if isinstance(col_names, list):
                tmp_col_names = col_names
            else:
                tmp_col_names = [col_names]

        return self.implementation.add_internal_vertex_id(
            tmp_df, id_column_name, tmp_col_names, (drop and can_drop),
            preserve_order
        )

    def from_internal_vertex_id(
        self,
        df,
        internal_column_name=None,
        external_column_names=None,
        drop=False,
    ):
        """
        Given a collection of internal vertex ids, return a DataFrame of
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

        internal_column_name: (optional) string
            Name of the column containing the internal vertex id.
            If df is a series then this parameter is ignored.  If df is
            a DataFrame this parameter is required.

        external_column_names: (optional) string or list of strings
            Name of the columns that define an external vertex id.
            If not specified, columns will be labeled '0', '1,', ..., 'n-1'

        drop: (optional) boolean
            If True the internal column name will be dropped from the
            DataFrame.  Defaults to False.

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original DataFrame columns exist unmodified.  Columns
            are added to the DataFrame to identify the external vertex
            identifiers. If external_columns is specified, these names
            are used as the names of the output columns.  If external_columns
            is not specifed the columns are labeled '0', ... 'n-1' based on
            the number of columns identifying the external vertex identifiers.
        """
        tmp_df = None
        can_drop = True
        if type(df) is cudf.Series:
            tmp_df = df.to_frame("id")
            internal_column_name = "id"
            can_drop = False
        elif type(df) is dask_cudf.Series:
            tmp_df = df.to_frame("id")
            internal_column_name = "id"
            can_drop = False
        else:
            tmp_df = df

        output_df = self.implementation.from_internal_vertex_id(
            tmp_df, internal_column_name, external_column_names
        )

        if drop and can_drop:
            return output_df.drop(columns=internal_column_name)

        return output_df

    def column_names(self):
        """
        Return the list of internal column names

        Returns
        ----------
            List of column names ('0', '1', ..., 'n-1')
        """
        return self.implementation.col_names

    def renumber(df, src_col_names, dst_col_names, preserve_order=False,
                 store_transposed=False):
        """
        Given a single GPU or distributed DataFrame, use src_col_names and
        dst_col_names to identify the source vertex identifiers and destination
        vertex identifiers, respectively.

        Internal vertex identifiers will be created, numbering vertices as
        integers starting from 0.

        The function will return a DataFrame containing the original dataframe
        contents with a new column labeled 'src' containing the renumbered
        source vertices and a new column labeled 'dst' containing the
        renumbered dest vertices, along with a NumberMap object that contains
        the number map for the numbering that was used.

        Note that this function does not guarantee order in single GPU mode,
        and does not guarantee order or partitioning in multi-GPU mode.  If you
        wish to preserve ordering, add an index column to df and sort the
        return by that index column.

        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            Contains a list of external vertex identifiers that will be
            numbered by the NumberMap class.
        src_col_names: string or list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier for source vertices
        dst_col_names: string or list of strings
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier for destination vertices
        store_transposed : bool
            Identify how the graph adjacency will be used.
            If True, the graph will be organized by destination.
            If False, the graph will be organized by source

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original DataFrame columns exist unmodified.  Columns
            are added to the DataFrame to identify the external vertex
            identifiers. If external_columns is specified, these names
            are used as the names of the output columns.  If external_columns
            is not specifed the columns are labeled '0', ... 'n-1' based on
            the number of columns identifying the external vertex identifiers.

        number_map : NumberMap
            The number map object object that retains the mapping between
            internal vertex identifiers and external vertex identifiers.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>>
        >>> df, number_map = NumberMap.renumber(df, '0', '1')
        >>>
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(df, 'src', 'dst')
        """
        renumber_map = NumberMap()

        if isinstance(src_col_names, list):
            renumber_map.from_dataframe(df, src_col_names, dst_col_names)
            df = renumber_map.add_internal_vertex_id(
                df, "src", src_col_names, drop=True,
                preserve_order=preserve_order
            )
            df = renumber_map.add_internal_vertex_id(
                df, "dst", dst_col_names, drop=True,
                preserve_order=preserve_order
            )
        else:
            renumber_map.from_dataframe(df, [src_col_names], [dst_col_names])
            df = renumber_map.add_internal_vertex_id(
                df, "src", src_col_names, drop=True,
                preserve_order=preserve_order
            )

            df = renumber_map.add_internal_vertex_id(
                df, "dst", dst_col_names, drop=True,
                preserve_order=preserve_order
            )

        if type(df) is dask_cudf.DataFrame:
            df = df.persist()

        return df, renumber_map

    def unrenumber(self, df, column_name, preserve_order=False):
        """
        Given a DataFrame containing internal vertex ids in the identified
        column, replace this with external vertex ids.  If the renumbering
        is from a single column, the output dataframe will use the same
        name for the external vertex identifiers.  If the renumbering is from
        a multi-column input, the output columns will be labeled 0 through
        n-1 with a suffix of _column_name.

        Note that this function does not guarantee order or partitioning in
        multi-GPU mode.

        Parameters
        ----------
        df: cudf.DataFrame or dask_cudf.DataFrame
            A DataFrame containing internal vertex identifiers that will be
            converted into external vertex identifiers.

        column_name: string
            Name of the column containing the internal vertex id.

        preserve_order: (optional) bool
            If True, preserve the ourder of the rows in the output
            DataFrame to match the input DataFrame

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original DataFrame columns exist unmodified.  The external
            vertex identifiers are added to the DataFrame, the internal
            vertex identifier column is removed from the dataframe.

        Examples
        --------
        >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
        >>>                   dtype=['int32', 'int32', 'float32'], header=None)
        >>>
        >>> df, number_map = NumberMap.renumber(df, '0', '1')
        >>>
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(df, 'src', 'dst')
        >>>
        >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500,
        >>>                       tol = 1.0e-05)
        >>>
        >>> pr = number_map.unrenumber(pr, 'vertex')
        >>>
        """
        if len(self.implementation.col_names) == 1:
            # Output will be renamed to match input
            mapping = {"0": column_name}
        else:
            # Output will be renamed to ${i}_${column_name}
            mapping = {}
            for nm in self.implementation.col_names:
                mapping[nm] = nm + "_" + column_name

        if preserve_order:
            index_name = NumberMap.generate_unused_column_name(df)
            df[index_name] = df.index

        df = self.from_internal_vertex_id(df, column_name, drop=True)

        if preserve_order:
            df = df.sort_values(
                index_name
            ).drop(index_name).reset_index(drop=True)

        if type(df) is dask_cudf.DataFrame:
            return df.map_partitions(
                lambda df: df.rename(columns=mapping, copy=False)
            )
        else:
            return df.rename(columns=mapping, copy=False)
