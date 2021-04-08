# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
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
#

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.structure import renumber_wrapper as c_renumber
from cugraph.utilities.utils import is_device_version_less_than
import cugraph.comms.comms as Comms
import dask_cudf
import numpy as np
import cudf


def call_renumber(sID,
                  data,
                  num_edges,
                  is_mnmg,
                  store_transposed):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return c_renumber.renumber(data[0],
                               num_edges,
                               wid,
                               handle,
                               is_mnmg,
                               store_transposed)


class NumberMap:

    class SingleGPU:
        def __init__(self, df, src_col_names, dst_col_names, id_type,
                     store_transposed):
            self.col_names = NumberMap.compute_vals(src_col_names)
            self.src_col_names = src_col_names
            self.dst_col_names = dst_col_names
            self.df = df
            self.id_type = id_type
            self.store_transposed = store_transposed
            self.numbered = False

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

        def indirection_map(self, df, src_col_names, dst_col_names):
            tmp_df = cudf.DataFrame()

            tmp = (
                df[src_col_names]
                .groupby(src_col_names)
                .count()
                .reset_index()
                .rename(
                    columns=dict(zip(src_col_names, self.col_names)),
                    copy=False,
                )
            )

            if dst_col_names is not None:
                tmp_dst = (
                    df[dst_col_names]
                    .groupby(dst_col_names)
                    .count()
                    .reset_index()
                )
                for newname, oldname in zip(self.col_names, dst_col_names):
                    tmp_df[newname] = tmp[newname].append(tmp_dst[oldname])
            else:
                for newname in self.col_names:
                    tmp_df[newname] = tmp[newname]

            tmp_df = tmp_df.groupby(self.col_names).count().reset_index()
            tmp_df["id"] = tmp_df.index.astype(self.id_type)
            self.df = tmp_df
            return tmp_df

    class MultiGPU:
        def __init__(
            self, ddf, src_col_names, dst_col_names, id_type, store_transposed
        ):
            self.col_names = NumberMap.compute_vals(src_col_names)
            self.val_types = NumberMap.compute_vals_types(ddf, src_col_names)
            self.val_types["count"] = np.int32
            self.id_type = id_type
            self.ddf = ddf
            self.store_transposed = store_transposed
            self.numbered = False

        def to_internal_vertex_id(self, ddf, col_names):
            return self.ddf.merge(
                ddf,
                right_on=col_names,
                left_on=self.col_names,
                how="right",
            )["global_id"]

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

        def indirection_map(self, ddf, src_col_names, dst_col_names):

            tmp = (
                ddf[src_col_names]
                .groupby(src_col_names)
                .count()
                .reset_index()
                .rename(
                    columns=dict(zip(src_col_names, self.col_names)),
                )
            )

            if dst_col_names is not None:
                tmp_dst = (
                    ddf[dst_col_names]
                    .groupby(dst_col_names)
                    .count()
                    .reset_index()
                )
                for i, (newname, oldname) in enumerate(zip(self.col_names,
                                                           dst_col_names)):
                    if i == 0:
                        tmp_df = tmp[newname].append(tmp_dst[oldname]).\
                            to_frame(name=newname)
                    else:
                        tmp_df[newname] = tmp[newname].append(tmp_dst[oldname])
            else:
                for newname in self.col_names:
                    tmp_df[newname] = tmp[newname]
            tmp_ddf = tmp_df.groupby(self.col_names).count().reset_index()

            # Set global index
            tmp_ddf = tmp_ddf.assign(idx=1)
            tmp_ddf['global_id'] = tmp_ddf.idx.cumsum() - 1
            tmp_ddf = tmp_ddf.drop(columns='idx')
            tmp_ddf = tmp_ddf.persist()
            self.ddf = tmp_ddf
            return tmp_ddf

    def __init__(self, id_type=np.int32):
        self.implementation = None
        self.id_type = id_type

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

    def renumber(df, src_col_names, dst_col_names, preserve_order=False,
                 store_transposed=False):
        if isinstance(src_col_names, list):
            renumber_type = 'legacy'
        elif not (df[src_col_names].dtype == np.int32 or
                  df[src_col_names].dtype == np.int64):
            renumber_type = 'legacy'
        elif is_device_version_less_than((7, 0)):
            renumber_type = 'legacy'
        else:
            renumber_type = 'experimental'

        renumber_map = NumberMap()
        if not isinstance(src_col_names, list):
            src_col_names = [src_col_names]
            dst_col_names = [dst_col_names]
        if type(df) is cudf.DataFrame:
            renumber_map.implementation = NumberMap.SingleGPU(
                df, src_col_names, dst_col_names, renumber_map.id_type,
                store_transposed
            )
        elif type(df) is dask_cudf.DataFrame:
            renumber_map.implementation = NumberMap.MultiGPU(
                df, src_col_names, dst_col_names, renumber_map.id_type,
                store_transposed
            )
        else:
            raise Exception("df must be cudf.DataFrame or dask_cudf.DataFrame")

        if renumber_type == 'legacy':
            indirection_map = renumber_map.implementation.\
                              indirection_map(df,
                                              src_col_names,
                                              dst_col_names)
            df = renumber_map.add_internal_vertex_id(
                df, "src", src_col_names, drop=True,
                preserve_order=preserve_order
            )
            df = renumber_map.add_internal_vertex_id(
                df, "dst", dst_col_names, drop=True,
                preserve_order=preserve_order
            )
        else:
            df = df.rename(columns={src_col_names[0]: "src",
                                    dst_col_names[0]: "dst"})

        num_edges = len(df)

        if isinstance(df, dask_cudf.DataFrame):
            is_mnmg = True
        else:
            is_mnmg = False

        if is_mnmg:
            client = default_client()
            data = get_distributed_data(df)
            result = [(client.submit(call_renumber,
                                     Comms.get_session_id(),
                                     wf[1],
                                     num_edges,
                                     is_mnmg,
                                     store_transposed,
                                     workers=[wf[0]]), wf[0])
                      for idx, wf in enumerate(data.worker_to_parts.items())]
            wait(result)

            def get_renumber_map(data):
                return data[0]

            def get_renumbered_df(data):
                return data[1]

            renumbering_map = dask_cudf.from_delayed(
                                 [client.submit(get_renumber_map,
                                                data,
                                                workers=[wf])
                                     for (data, wf) in result])
            renumbered_df = dask_cudf.from_delayed(
                               [client.submit(get_renumbered_df,
                                              data,
                                              workers=[wf])
                                   for (data, wf) in result])
            if renumber_type == 'legacy':
                renumber_map.implementation.ddf = indirection_map.merge(
                    renumbering_map,
                    right_on='original_ids', left_on='global_id',
                    how='right').\
                    drop(columns=['global_id', 'original_ids'])\
                    .rename(columns={'new_ids': 'global_id'})
            else:
                renumber_map.implementation.ddf = renumbering_map.rename(
                    columns={'original_ids': '0', 'new_ids': 'global_id'})
            renumber_map.implementation.numbered = True
            return renumbered_df, renumber_map

        else:
            if is_device_version_less_than((7, 0)):
                renumbered_df = df
                renumber_map.implementation.df = indirection_map
                renumber_map.implementation.numbered = True
                return renumbered_df, renumber_map

            renumbering_map, renumbered_df = c_renumber.renumber(
                                             df,
                                             num_edges,
                                             0,
                                             Comms.get_default_handle(),
                                             is_mnmg,
                                             store_transposed)
            if renumber_type == 'legacy':
                renumber_map.implementation.df = indirection_map.\
                    merge(renumbering_map,
                          right_on='original_ids', left_on='id').\
                    drop(columns=['id', 'original_ids'])\
                    .rename(columns={'new_ids': 'id'}, copy=False)
            else:
                renumber_map.implementation.df = renumbering_map.rename(
                    columns={'original_ids': '0', 'new_ids': 'id'}, copy=False)

            renumber_map.implementation.numbered = True
            return renumbered_df, renumber_map

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
            ).drop(columns=index_name).reset_index(drop=True)

        if type(df) is dask_cudf.DataFrame:
            return df.map_partitions(
                lambda df: df.rename(columns=mapping, copy=False)
            )
        else:
            return df.rename(columns=mapping, copy=False)
