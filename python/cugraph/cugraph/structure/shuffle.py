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

from dask.dataframe.shuffle import rearrange_by_column
import cudf
import cugraph.comms.comms as Comms


def _set_partitions_pre(df, vertex_row_partitions, vertex_col_partitions,
                        prows, pcols, transposed, partition_type):
    if transposed:
        r = df['dst']
        c = df['src']
    else:
        r = df['src']
        c = df['dst']
    r_div = vertex_row_partitions.searchsorted(r, side='right')-1
    c_div = vertex_col_partitions.searchsorted(c, side='right')-1

    if partition_type == 1:
        partitions = r_div * pcols + c_div
    else:
        partitions = r_div % prows + c_div * prows
    return partitions


def shuffle(dg, transposed=False):
    """
    Shuffles the renumbered input distributed graph edgelist into ngpu
    partitions. The number of processes/gpus P = prows*pcols. The 2D
    partitioning divides the matrix into P*pcols rectangular partitions
    as per vertex partitioning performed in renumbering, and then shuffles
    these partitions into P gpus.
    """

    ddf = dg.edgelist.edgelist_df
    ngpus = Comms.get_n_workers()
    prows, pcols, partition_type = Comms.get_2D_partition()

    renumber_vertex_count = dg.renumber_map.implementation.\
        ddf.map_partitions(len).compute()
    renumber_vertex_cumsum = renumber_vertex_count.cumsum()

    if transposed:
        row_dtype = ddf['dst'].dtype
        col_dtype = ddf['src'].dtype
    else:
        row_dtype = ddf['src'].dtype
        col_dtype = ddf['dst'].dtype

    vertex_partition_offsets = cudf.Series([0], dtype=row_dtype)
    vertex_partition_offsets = vertex_partition_offsets.append(cudf.Series(
        renumber_vertex_cumsum, dtype=row_dtype))
    num_verts = vertex_partition_offsets.iloc[-1]
    if partition_type == 1:
        vertex_row_partitions = []
        for i in range(prows + 1):
            vertex_row_partitions.append(
                vertex_partition_offsets.iloc[i*pcols])
        vertex_row_partitions = cudf.Series(
            vertex_row_partitions, dtype=row_dtype)
    else:
        vertex_row_partitions = vertex_partition_offsets
    vertex_col_partitions = []
    for i in range(pcols + 1):
        vertex_col_partitions.append(vertex_partition_offsets.iloc[i*prows])
    vertex_col_partitions = cudf.Series(vertex_col_partitions, dtype=col_dtype)

    meta = ddf._meta._constructor_sliced([0])
    partitions = ddf.map_partitions(
        _set_partitions_pre,
        vertex_row_partitions=vertex_row_partitions,
        vertex_col_partitions=vertex_col_partitions, prows=prows,
        pcols=pcols, transposed=transposed, partition_type=partition_type,
        meta=meta)
    ddf2 = ddf.assign(_partitions=partitions)
    ddf3 = rearrange_by_column(
        ddf2,
        "_partitions",
        max_branch=None,
        npartitions=ngpus,
        shuffle="tasks",
        ignore_index=True,
    ).drop(columns=["_partitions"])

    partition_row_size = pcols
    partition_col_size = prows

    return (ddf3,
            num_verts,
            partition_row_size,
            partition_col_size,
            vertex_partition_offsets)
