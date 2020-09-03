import math
from dask.dataframe.shuffle import rearrange_by_column


def get_n_workers():
    from dask.distributed import default_client
    client = default_client()
    return len(client.scheduler_info()['workers'])


def get_2D_div(ngpus):
    pcols = int(math.sqrt(ngpus))
    while ngpus % pcols != 0:
        pcols = pcols - 1
    return int(ngpus/pcols), pcols


def _set_partitions_pre(df, num_verts, prows, pcols):
    rows_per_div = math.ceil(num_verts/prows)
    cols_per_div = math.ceil(num_verts/pcols)
    partitions = df['src'].floordiv(rows_per_div) * pcols + df['dst'].floordiv(cols_per_div)
    return partitions


def shuffle(dg):
    ddf = dg.edgelist.edgelist_df
    ngpus = get_n_workers()
    prows, pcols = get_2D_div(ngpus)
    num_verts = dg.number_of_nodes()

    meta = ddf._meta._constructor_sliced([0])
    partitions = ddf.map_partitions(
    _set_partitions_pre, num_verts, prows, pcols, meta=meta
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

