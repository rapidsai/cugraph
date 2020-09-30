import math
from dask.dataframe.shuffle import rearrange_by_column
import cudf


def get_n_workers():
    from dask.distributed import default_client
    client = default_client()
    return len(client.scheduler_info()['workers'])


def get_2D_div(ngpus):
    pcols = int(math.sqrt(ngpus))
    while ngpus % pcols != 0:
        pcols = pcols - 1
    return int(ngpus/pcols), pcols


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


def shuffle(dg, transposed=False, prows=None, pcols=None, partition_type=1):
    """
    Shuffles the renumbered input distributed graph edgelist into ngpu
    partitions. The number of processes/gpus P = prows*pcols. The 2D
    partitioning divides the matrix into P*pcols rectangular partitions
    as per vertex partitioning performed in renumbering, and then shuffles
    these partitions into P gpus.
    """

    ddf = dg.edgelist.edgelist_df
    ngpus = get_n_workers()
    if prows is None and pcols is None:
        if partition_type == 1:
            pcols, prows = get_2D_div(ngpus)
        else:
            prows, pcols = get_2D_div(ngpus)
    else:
        if prows is not None and pcols is not None:
            if ngpus != prows*pcols:
                raise Exception('prows*pcols should be equal to the\
 number of processes')
        elif prows is not None:
            if ngpus % prows != 0:
                raise Exception('prows must be a factor of the number\
 of processes')
            pcols = int(ngpus/prows)
        elif pcols is not None:
            if ngpus % pcols != 0:
                raise Exception('pcols must be a factor of the number\
 of processes')
            prows = int(ngpus/pcols)

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

    return ddf3, num_verts, vertex_partition_offsets
