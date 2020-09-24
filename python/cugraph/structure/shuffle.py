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


def _set_partitions_pre(df, vertex_row_partitions, vertex_col_partitions, prows, pcols):
    src_div = vertex_row_partitions.searchsorted(df['src'], side='right')-1
    dst_div = vertex_col_partitions.searchsorted(df['dst'], side='right')-1
    partitions = src_div%prows + dst_div*prows
    return partitions


def shuffle(dg):
    ddf = dg.edgelist.edgelist_df
    ngpus = get_n_workers()
    prows, pcols = get_2D_div(ngpus)

    renumber_vertex_count = dg.renumber_map.implementation.ddf.map_partitions(len).compute()
    renumber_vertex_cumsum = renumber_vertex_count.cumsum()
    src_dtype = ddf['src'].dtype
    dst_dtype = ddf['dst'].dtype

    vertex_row_partitions = cudf.Series([0], dtype=src_dtype)
    vertex_row_partitions = vertex_row_partitions.append(cudf.Series(renumber_vertex_cumsum, dtype = src_dtype))
    num_verts = vertex_row_partitions.iloc[-1]
    vertex_col_partitions = []
    for i in range(pcols + 1):
        vertex_col_partitions.append(vertex_row_partitions.iloc[i*prows])
    vertex_col_partitions = cudf.Series(vertex_col_partitions, dtype = dst_dtype)

    meta = ddf._meta._constructor_sliced([0])
    partitions = ddf.map_partitions(
    _set_partitions_pre, vertex_row_partitions=vertex_row_partitions, vertex_col_partitions=vertex_col_partitions,prows= prows, pcols=pcols, meta=meta
    )
    ddf2 = ddf.assign(_partitions = partitions)
    ddf3 = rearrange_by_column(
        ddf2,
        "_partitions",
        max_branch=None,
        npartitions=ngpus,
        shuffle="tasks",
        ignore_index=True,
    ).drop(columns=["_partitions"])

    return ddf3
