# Copyright (c) 2021, NVIDIA CORPORATION.
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
import cugraph.comms as Comms
import dask_cudf
import numpy as np
import cudf
import cugraph.structure.number_map as legacy_number_map


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
            self.df = cudf.DataFrame()
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

    class MultiGPU:
        def __init__(
            self, ddf, src_col_names, dst_col_names, id_type, store_transposed
        ):
            self.col_names = NumberMap.compute_vals(src_col_names)
            self.val_types = NumberMap.compute_vals_types(ddf, src_col_names)
            self.val_types["count"] = np.int32
            self.id_type = id_type
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

    def renumber(df, src_col_names, dst_col_names, preserve_order=False,
                 store_transposed=False):

        if isinstance(src_col_names, list):
            renumber_type = 'legacy'
        # elif isinstance(df[src_col_names].dtype, string):
        #    renumber_type = 'legacy'
        else:
            renumber_type = 'experimental'

        if renumber_type == 'legacy':
            renumber_map, renumbered_df = legacy_number_map.renumber(
                                              df,
                                              src_col_names,
                                              dst_col_names,
                                              preserve_order,
                                              store_transposed)
            # Add shuffling once algorithms are switched to new renumber
            # (ddf,
            # num_verts,
            # partition_row_size,
            # partition_col_size,
            # vertex_partition_offsets) = shuffle(input_graph, transposed=True)
            return renumber_map, renumbered_df

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

            renumber_map.implementation.ddf = renumbering_map
            renumber_map.implementation.numbered = True

            return renumbered_df, renumber_map
        else:
            renumbering_map, renumbered_df = c_renumber.renumber(
                                             df,
                                             num_edges,
                                             0,
                                             Comms.get_default_handle(),
                                             is_mnmg,
                                             store_transposed)
            renumber_map.implementation.df = renumbering_map
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
        if len(self.col_names) == 1:
            # Output will be renamed to match input
            mapping = {"0": column_name}
        else:
            # Output will be renamed to ${i}_${column_name}
            mapping = {}
            for nm in self.col_names:
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
