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
                        prows, pcols, transposed):
    if transposed:
        r = df['dst']
        c = df['src']
    else:
        r = df['src']
        c = df['dst']
    r_div = vertex_row_partitions.searchsorted(r, side='right')-1
    c_div = vertex_col_partitions.searchsorted(c, side='right')-1
    partitions = r_div % prows + c_div * prows
    return partitions


def shuffle(dg, transposed=False, prows=None, pcols=None):
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
    src_dtype = ddf['src'].dtype
    dst_dtype = ddf['dst'].dtype

    vertex_row_partitions = cudf.Series([0], dtype=src_dtype)
    vertex_row_partitions = vertex_row_partitions.append(cudf.Series(
        renumber_vertex_cumsum, dtype=src_dtype))
    num_verts = vertex_row_partitions.iloc[-1]
    vertex_col_partitions = []
    for i in range(pcols + 1):
        vertex_col_partitions.append(vertex_row_partitions.iloc[i*prows])
    vertex_col_partitions = cudf.Series(vertex_col_partitions, dtype=dst_dtype)

    meta = ddf._meta._constructor_sliced([0])
    partitions = ddf.map_partitions(
        _set_partitions_pre,
        vertex_row_partitions=vertex_row_partitions,
        vertex_col_partitions=vertex_col_partitions, prows=prows,
        pcols=pcols, transposed=transposed, meta=meta)
    ddf2 = ddf.assign(_partitions=partitions)
    ddf3 = rearrange_by_column(
        ddf2,
        "_partitions",
        max_branch=None,
        npartitions=ngpus,
        shuffle="tasks",
        ignore_index=True,
    ).drop(columns=["_partitions"])

    return ddf3, num_verts, vertex_row_partitions
