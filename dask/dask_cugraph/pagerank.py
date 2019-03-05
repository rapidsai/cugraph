import cugraph 
import cudf
import dask_cudf as dc
import dask.dataframe as dd


def _mg_pagerank(data, global_v):
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #print(data)
    pr_col_length = data[data.columns[2]][len(data)-1] - data[data.columns[2]][0] + 1
    pr_df = cugraph.mg_pagerank(data, pr_col_length, global_v)
    return pr_df

    
def get_rank(l):
    from mpi4py import MPI    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    return rank

    
def _print_data(data):
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("RANK:",rank)
    print("DATA:")
    print(data)


def mg_pagerank(X_df):
    client = default_client()
    gpu_futures = _get_mg_info(X_df)
    ddf = dd.from_delayed(gpu_futures)
    #gpu_futures = print_data(z)   
    return ddf


import logging
from dask.distributed import Client, wait, default_client, futures_of
import dask.dataframe as dd
from toolz import first, assoc


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def _get_mg_info(ddf):
    client = default_client()
    if isinstance(ddf, dd.DataFrame):
        parts = ddf.to_delayed()
        parts = client.compute(parts)
        wait(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = client.who_has(parts)

    x = list(client.has_what().keys())
    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    ll=[1,1,1,1]
    worker_ranks = [(client.submit(get_rank, p,workers=[worker]).result(), worker) for p,worker in zip(parts,x)]
    rank_to_worker_dict = dict(worker_ranks)
    parts_to_worker_map = []
    for i,part in enumerate(parts):
        parts_to_worker_map.append((part, rank_to_worker_dict[i]))


    max_node_src_col = ddf[ddf.columns[1]].max().compute()
    max_node_dest_col = ddf[ddf.columns[2]].max().compute()
    num_vertices = max(max_node_src_col,max_node_dest_col) + 1

    gpu_futures = [client.submit(_mg_pagerank, part, num_vertices, workers=[worker]) for part, worker in parts_to_worker_map]
    wait(gpu_futures)

    return gpu_futures


def print_data(ddf):
    client = default_client()
    
    parts = ddf.to_delayed()
    parts = client.compute(parts)
    wait(parts)
    print(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = client.who_has(parts)
    
    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))
    
    gpu_futures = [client.submit(_print_data, part,workers=[worker]) for worker, part in worker_map]
    wait(gpu_futures)

    return gpu_futures


