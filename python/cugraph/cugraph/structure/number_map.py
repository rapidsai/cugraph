# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

from collections.abc import Iterable

from dask.distributed import wait, default_client
import dask_cudf
import numpy as np
import cudf

from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.structure import renumber_wrapper as c_renumber
import cugraph.dask.comms.comms as Comms


def call_renumber(
    sID,
    data,
    renumbered_src_col_name,
    renumbered_dst_col_name,
    num_edges,
    is_mnmg,
    store_transposed,
):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return c_renumber.renumber(
        data[0],
        renumbered_src_col_name,
        renumbered_dst_col_name,
        num_edges,
        wid,
        handle,
        is_mnmg,
        store_transposed,
    )


class NumberMap:
    class SingleGPU:
        def __init__(self, df, src_col_names, dst_col_names, id_type, store_transposed):
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

        def add_internal_vertex_id(
            self, df, id_column_name, col_names, drop, preserve_order
        ):
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
                ret = merge_df.merge(
                    tmp_df,
                    right_on=col_names,
                    left_on=self.col_names,
                    how="right",
                ).drop(columns=self.col_names)

            if drop:
                ret = ret.drop(columns=col_names)

            ret = ret.rename(columns={id_name: id_column_name}, copy=False)

            if preserve_order:
                ret = ret.sort_values(index_name).reset_index(drop=True)

            return ret

        def indirection_map(self, df, src_col_names, dst_col_names):
            # src_col_names and dst_col_names are lists
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
                tmp_dst = df[dst_col_names].groupby(dst_col_names).count().reset_index()
                # Need to have the same column names before both df can be
                # concat
                tmp_dst.columns = tmp.columns
                tmp_df = cudf.concat([tmp, tmp_dst])
            else:
                newname = self.col_names
                tmp_df = tmp[newname]

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
            tmp_ddf = ddf[col_names].rename(
                columns=dict(zip(col_names, self.col_names))
            )
            for name in self.col_names:
                tmp_ddf[name] = tmp_ddf[name].astype(self.ddf[name].dtype)
            x = self.ddf.merge(
                tmp_ddf,
                on=self.col_names,
                how="right",
            )
            return x["global_id"]

        def from_internal_vertex_id(
            self, df, internal_column_name, external_column_names
        ):
            tmp_df = self.ddf.merge(
                df, right_on=internal_column_name, left_on="global_id", how="right"
            ).map_partitions(lambda df: df.drop(columns="global_id"))

            if external_column_names is None:
                return tmp_df
            else:
                return tmp_df.map_partitions(
                    lambda df: df.rename(
                        columns=dict(zip(self.col_names, external_column_names)),
                        copy=False,
                    )
                )

        def add_internal_vertex_id(
            self, ddf, id_column_name, col_names, drop, preserve_order
        ):
            # At the moment, preserve_order cannot be done on
            # multi-GPU
            if preserve_order:
                raise Exception("preserve_order not supported for multi-GPU")

            ret = None
            if col_names is None:
                ret = self.ddf.merge(ddf, on=self.col_names, how="right")
            elif col_names == self.col_names:
                ret = self.ddf.merge(ddf, on=col_names, how="right")
            else:
                ret = self.ddf.merge(
                    ddf, right_on=col_names, left_on=self.col_names
                ).map_partitions(lambda df: df.drop(columns=self.col_names))

            if drop:
                ret = ret.map_partitions(lambda df: df.drop(columns=col_names))

            ret = ret.map_partitions(
                lambda df: df.rename(columns={"global_id": id_column_name}, copy=False)
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
                    ddf[dst_col_names].groupby(dst_col_names).count().reset_index()
                )
                tmp_dst.columns = tmp.columns
                tmp_df = dask_cudf.concat([tmp, tmp_dst])

            else:
                newname = self.col_names
                tmp_df = tmp[newname]
            tmp_ddf = tmp_df.groupby(self.col_names).count().reset_index()

            # Set global index
            tmp_ddf = tmp_ddf.assign(idx=1)
            # ensure the original vertex and the 'global_id' columns are
            # of the same type unless the original vertex type is 'string'
            tmp_ddf["global_id"] = tmp_ddf.idx.cumsum().astype(self.id_type) - 1
            tmp_ddf = tmp_ddf.drop(columns="idx")
            tmp_ddf = tmp_ddf.persist()
            self.ddf = tmp_ddf
            return tmp_ddf

    def __init__(self, renumber_id_type=np.int32, unrenumbered_id_type=np.int32):
        self.implementation = None
        self.renumber_id_type = renumber_id_type
        self.unrenumbered_id_type = unrenumbered_id_type
        # The default src/dst column names in the resulting renumbered
        # dataframe. These may be updated by the renumbering methods if the
        # input dataframe uses the default names.
        self.renumbered_src_col_name = "renumbered_src"
        self.renumbered_dst_col_name = "renumbered_dst"

    @staticmethod
    def compute_vals_types(df, column_names):
        """
        Helper function to compute internal column names and types
        """
        return {str(i): df[column_names[i]].dtype for i in range(len(column_names))}

    @staticmethod
    def generate_unused_column_name(column_names, start_with_name="col"):
        """
        Helper function to generate an unused column name
        """
        name = start_with_name
        counter = 2
        while name in column_names:
            name = f"{start_with_name}{counter}"
            counter += 1
        return name

    @staticmethod
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
            tmp_df = df.to_frame()
            tmp_col_names = tmp_df.columns
        else:
            tmp_df = df
            tmp_col_names = col_names

        reply = self.implementation.to_internal_vertex_id(tmp_df, tmp_col_names)
        return reply

    def add_internal_vertex_id(
        self, df, id_column_name="id", col_names=None, drop=False, preserve_order=False
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

        id_column_name: string, optional (default="id")
            The name to be applied to the column containing the id

        col_names: list of strings, optional (default=None)
            This list of 1 or more strings contain the names
            of the columns that uniquely identify an external
            vertex identifier

        drop: boolean, optional (default=False)
            If True, drop the column names specified in col_names from
            the returned DataFrame.

        preserve_order: boolean, optional (default=False)
            If True, do extra sorting work to preserve the order
            of the input DataFrame.

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
            tmp_df, id_column_name, tmp_col_names, (drop and can_drop), preserve_order
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

        internal_column_name: string, optional (default=None)
            Name of the column containing the internal vertex id.
            If df is a series then this parameter is ignored.  If df is
            a DataFrame this parameter is required.

        external_column_names: string or list of str, optional (default=None)
            Name of the columns that define an external vertex id.
            If not specified, columns will be labeled '0', '1,', ..., 'n-1'

        drop: boolean, optional (default=False)
            If True the internal column name will be dropped from the
            DataFrame.

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

    @staticmethod
    def renumber_and_segment(
        df,
        src_col_names,
        dst_col_names,
        preserve_order=False,
        store_transposed=False,
        legacy_renum_only=False,
    ):
        renumbered = True

        # For columns with mismatch dtypes, set the renumbered
        # id_type to either 'int32' or 'int64'
        if df.dtypes.nunique() > 1:
            # can't determine the edgelist input type
            unrenumbered_id_type = None
        else:
            unrenumbered_id_type = df.dtypes[0]

        if np.int64 in list(df.dtypes):
            renumber_id_type = np.int64
        else:
            # renumber the edgelist to 'int32'
            renumber_id_type = np.int32

        # FIXME: Drop the renumber_type 'experimental' once all the
        # algos follow the C/Pylibcugraph path

        # The renumber_type 'legacy' runs both the python and the
        # C++ renumbering.
        if isinstance(src_col_names, list):
            renumber_type = "legacy"

        elif not (
            df[src_col_names].dtype == np.int32 or df[src_col_names].dtype == np.int64
        ):
            renumber_type = "legacy"
        else:
            # The renumber_type 'experimental' only runs the C++
            # renumbering
            renumber_type = "experimental"

        if legacy_renum_only and renumber_type == "experimental":
            # The original dataframe will be returned.
            renumber_type = "skip_renumbering"
            renumbered = False

        renumber_map = NumberMap(renumber_id_type, unrenumbered_id_type)
        if not isinstance(src_col_names, list):
            src_col_names = [src_col_names]
            dst_col_names = [dst_col_names]

        # Assign the new src and dst column names to be used in the renumbered
        # dataframe to return (renumbered_src_col_name and
        # renumbered_dst_col_name)
        renumber_map.set_renumbered_col_names(src_col_names, dst_col_names, df.columns)

        if isinstance(df, cudf.DataFrame):
            renumber_map.implementation = NumberMap.SingleGPU(
                df,
                src_col_names,
                dst_col_names,
                renumber_map.renumber_id_type,
                store_transposed,
            )
        elif isinstance(df, dask_cudf.DataFrame):
            renumber_map.implementation = NumberMap.MultiGPU(
                df,
                src_col_names,
                dst_col_names,
                renumber_map.renumber_id_type,
                store_transposed,
            )
        else:
            raise TypeError("df must be cudf.DataFrame or dask_cudf.DataFrame")

        renumber_map.implementation.numbered = renumbered

        if renumber_type == "legacy":
            indirection_map = renumber_map.implementation.indirection_map(
                df, src_col_names, dst_col_names
            )
            df = renumber_map.add_internal_vertex_id(
                df,
                renumber_map.renumbered_src_col_name,
                src_col_names,
                drop=True,
                preserve_order=preserve_order,
            )
            df = renumber_map.add_internal_vertex_id(
                df,
                renumber_map.renumbered_dst_col_name,
                dst_col_names,
                drop=True,
                preserve_order=preserve_order,
            )
        elif renumber_type == "skip_renumbering":
            # Update the renumbered source and destination column name
            # with the original input's source and destination name
            renumber_map.renumbered_src_col_name = src_col_names[0]
            renumber_map.renumbered_dst_col_name = dst_col_names[0]

        else:
            df = df.rename(
                columns={
                    src_col_names[0]: renumber_map.renumbered_src_col_name,
                    dst_col_names[0]: renumber_map.renumbered_dst_col_name,
                }
            )
        num_edges = len(df)

        if isinstance(df, dask_cudf.DataFrame):
            is_mnmg = True
        else:
            is_mnmg = False

        if is_mnmg:
            # Do not renumber the algos following the C/Pylibcugraph path
            if renumber_type in ["legacy", "experimental"]:
                client = default_client()
                data = get_distributed_data(df)
                result = [
                    (
                        client.submit(
                            call_renumber,
                            Comms.get_session_id(),
                            wf[1],
                            renumber_map.renumbered_src_col_name,
                            renumber_map.renumbered_dst_col_name,
                            num_edges,
                            is_mnmg,
                            store_transposed,
                            workers=[wf[0]],
                        ),
                        wf[0],
                    )
                    for idx, wf in enumerate(data.worker_to_parts.items())
                ]
                wait(result)

                def get_renumber_map(id_type, data):
                    return data[0].astype(id_type)

                def get_segment_offsets(data):
                    return data[1]

                def get_renumbered_df(id_type, data):
                    data[2][renumber_map.renumbered_src_col_name] = data[2][
                        renumber_map.renumbered_src_col_name
                    ].astype(id_type)
                    data[2][renumber_map.renumbered_dst_col_name] = data[2][
                        renumber_map.renumbered_dst_col_name
                    ].astype(id_type)
                    return data[2]

                id_type = df[renumber_map.renumbered_src_col_name].dtype
                renumbering_map = dask_cudf.from_delayed(
                    [
                        client.submit(get_renumber_map, id_type, data, workers=[wf])
                        for (data, wf) in result
                    ]
                )

                list_of_segment_offsets = client.gather(
                    [
                        client.submit(get_segment_offsets, data, workers=[wf])
                        for (data, wf) in result
                    ]
                )
                aggregate_segment_offsets = []
                for segment_offsets in list_of_segment_offsets:
                    aggregate_segment_offsets.extend(segment_offsets)

                renumbered_df = dask_cudf.from_delayed(
                    [
                        client.submit(get_renumbered_df, id_type, data, workers=[wf])
                        for (data, wf) in result
                    ]
                )

                if renumber_type == "legacy":
                    renumber_map.implementation.ddf = (
                        indirection_map.merge(
                            renumbering_map,
                            right_on="original_ids",
                            left_on="global_id",
                            how="right",
                        )
                        .drop(columns=["global_id", "original_ids"])
                        .rename(columns={"new_ids": "global_id"})
                    )
                else:
                    renumber_map.implementation.ddf = renumbering_map.rename(
                        columns={"original_ids": "0", "new_ids": "global_id"}
                    )
                return renumbered_df, renumber_map, aggregate_segment_offsets

            else:
                # There is no aggregate_segment_offsets since the
                # C++ renumbering is skipped
                return df, renumber_map, None

        else:
            # Do not renumber the algos following the C/Pylibcugraph path
            if renumber_type in ["legacy", "experimental"]:
                renumbering_map, segment_offsets, renumbered_df = c_renumber.renumber(
                    df,
                    renumber_map.renumbered_src_col_name,
                    renumber_map.renumbered_dst_col_name,
                    num_edges,
                    0,
                    Comms.get_default_handle(),
                    is_mnmg,
                    store_transposed,
                )
                if renumber_type == "legacy":
                    renumber_map.implementation.df = (
                        indirection_map.merge(
                            renumbering_map, right_on="original_ids", left_on="id"
                        )
                        .drop(columns=["id", "original_ids"])
                        .rename(columns={"new_ids": "id"}, copy=False)
                    )
                else:
                    renumber_map.implementation.df = renumbering_map.rename(
                        columns={"original_ids": "0", "new_ids": "id"}, copy=False
                    )

                return renumbered_df, renumber_map, segment_offsets
            else:
                # There is no aggregate_segment_offsets since the
                # C++ renumbering is skipped
                return df, renumber_map, None

    @staticmethod
    def renumber(
        df,
        src_col_names,
        dst_col_names,
        preserve_order=False,
        store_transposed=False,
        legacy_renum_only=False,
    ):
        return NumberMap.renumber_and_segment(
            df,
            src_col_names,
            dst_col_names,
            preserve_order,
            store_transposed,
            legacy_renum_only,
        )[0:2]

    def unrenumber(self, df, column_name, preserve_order=False, get_column_names=False):
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

        preserve_order: bool, optional (default=False)
            If True, preserve the ourder of the rows in the output
            DataFrame to match the input DataFrame

        get_column_names: bool, optional (default=False)
            If True, the unrenumbered column names are returned.

        Returns
        ---------
        df : cudf.DataFrame or dask_cudf.DataFrame
            The original DataFrame columns exist unmodified.  The external
            vertex identifiers are added to the DataFrame, the internal
            vertex identifier column is removed from the dataframe.

        column_names: string or list of strings
            If get_column_names is True, the unrenumbered column names are
            returned.

        Examples
        --------
        >>> from cugraph.structure import number_map
        >>> df = cudf.read_csv(datasets_path / 'karate.csv', delimiter=' ',
        ...                    dtype=['int32', 'int32', 'float32'],
        ...                    header=None)
        >>> df, number_map = number_map.NumberMap.renumber(df, '0', '1')
        >>> G = cugraph.Graph()
        >>> G.from_cudf_edgelist(df,
        ...                      number_map.renumbered_src_col_name,
        ...                      number_map.renumbered_dst_col_name)
        >>> pr = cugraph.pagerank(G, alpha = 0.85, max_iter = 500,
        ...                       tol = 1.0e-05)
        >>> pr = number_map.unrenumber(pr, 'vertex')

        """
        if len(self.implementation.col_names) == 1:
            # Output will be renamed to match input
            mapping = {"0": column_name}
            col_names = column_name
        else:
            # Output will be renamed to ${i}_${column_name}
            mapping = {}
            for nm in self.implementation.col_names:
                mapping[nm] = nm + "_" + column_name
            col_names = list(mapping.values())

        if preserve_order:
            index_name = NumberMap.generate_unused_column_name(df)
            df[index_name] = df.index

        df = self.from_internal_vertex_id(df, column_name, drop=True)

        if preserve_order:
            df = (
                df.sort_values(index_name)
                .drop(columns=index_name)
                .reset_index(drop=True)
            )

        if type(df) is dask_cudf.DataFrame:
            df = df.map_partitions(lambda df: df.rename(columns=mapping, copy=False))
        else:
            df = df.rename(columns=mapping, copy=False)
        if get_column_names:
            return df, col_names
        else:
            return df

    def vertex_column_size(self):
        return len(self.implementation.col_names)

    def set_renumbered_col_names(
        self, src_col_names_to_replace, dst_col_names_to_replace, all_col_names
    ):
        """
        Sets self.renumbered_src_col_name and self.renumbered_dst_col_name to
        values that can be used to replace src_col_names_to_replace and
        dst_col_names_to_replace to values that will not collide with any other
        column names in all_col_names.

        The new unique column names are generated using the existing
        self.renumbered_src_col_name and self.renumbered_dst_col_name as
        starting points.
        """
        assert isinstance(src_col_names_to_replace, Iterable)
        assert isinstance(dst_col_names_to_replace, Iterable)
        assert isinstance(all_col_names, Iterable)
        # No need to consider the col_names_to_replace when picking new unique
        # names, since those names will be replaced anyway, and replacing a
        # name with the same value is allowed.
        reserved_col_names = set(all_col_names) - set(
            src_col_names_to_replace + dst_col_names_to_replace
        )
        self.renumbered_src_col_name = self.generate_unused_column_name(
            reserved_col_names, start_with_name=self.renumbered_src_col_name
        )
        self.renumbered_dst_col_name = self.generate_unused_column_name(
            reserved_col_names, start_with_name=self.renumbered_dst_col_name
        )
