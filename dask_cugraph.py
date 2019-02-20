##DASK CODE
import random
import cugraph 
import cudf
import dask_cudf as dc
import dask.dataframe as dd

def _mg_pagerank(data, global_v):
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("RANK:",rank)
    print("DATA:")
    print(data)
    print(global_v)
    pr_col_length = data[data.columns[1]][len(data)-1] - data[data.columns[1]][0] + 1
    #print("Length: ",pr_col_length)
    #if rank is 0:
    #    pr_col_length = 14
    #if rank is 1:
    #    pr_col_length = 18
    pr_df = cugraph.mg_pagerank(data, pr_col_length, global_v)
    return pr_df
    
def _print_data(data):
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    print("RANK:",rank)
    print("DATA:")
    print(data)


def mg_pagerank(X_df):
    print("----------------> FUNTION NAME: MG_PAGERANK in dask_cugraph")
    
    client = default_client()
    gpu_futures = _get_mg_info(X_df)
    #print("GPU_FUTURES")
    #print(gpu_futures)
    z = dd.from_delayed(gpu_futures)
    print(z) 
    gpu_futures = print_data(z)   
    #print("GPU_FUTURES")
    #print(gpu_futures)
    return z

#from foo import print_data_and_rank
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
    print("----------------> FUNTION NAME: GET_MG_INFO")
    client = default_client()
    if isinstance(ddf, dd.DataFrame):
        parts = ddf.to_delayed()
        parts = client.compute(parts)
        wait(parts)
        print(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = client.who_has(parts)
    #print("WHO_HAS")
    #print(who_has)

    worker_map = []
    for key, workers in who_has.items():
        #print("workers")
        #print(workers)
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))
    max_node_src_col = ddf[ddf.columns[0]].max().compute()
    max_node_dest_col = ddf[ddf.columns[1]].max().compute()
    num_vertices = max(max_node_src_col,max_node_dest_col) + 1
    #print("VERTICE MAX ",num_vertices)
    gpu_futures = [client.submit(_mg_pagerank, part, num_vertices, workers=[worker]) for worker, part in worker_map]
    wait(gpu_futures)
    return gpu_futures


def _build_host_dict(gpu_futures, client):

    who_has = client.who_has(gpu_futures)
    print(who_has)
    key_to_host_dict = {}
    for key in who_has:
        key_to_host_dict[key] = parse_host_port(who_has[key][0])

    hosts_to_key_dict = {}
    for key, host in key_to_host_dict.items():
        if host not in hosts_to_key_dict:
            hosts_to_key_dict[host] = set([key])
        else:
            hosts_to_key_dict[host].add(key)

    workers = [key[0] for key in list(who_has.values())]
    return build_host_dict(workers)


############################# CORE.PY ##################################


def print_data(ddf):
    #print("----------------> FUNTION NAME: GET_MG_INFO")
    client = default_client()
    
    print("HERE$$$$$$$$$")
    parts = ddf.to_delayed()
    parts = client.compute(parts)
    wait(parts)
    print(parts)

    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = client.who_has(parts)
    print("WHO_HAS")
    print(who_has)

    worker_map = []
    for key, workers in who_has.items():
        print("workers")
        print(workers)
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))
    #gpu_futures = [(worker, client.submit(print_data_and_rank, part, workers=[worker])) for worker, part in worker_map]
    
    gpu_futures = [client.submit(_print_data, part,workers=[worker]) for worker, part in worker_map]
    #gpu_data = [(worker, client.submit(to_gpu_matrix, part, workers=[worker]))
    #                for worker, part in worker_map]    
    wait(gpu_futures)
    #wait(gpu_data)
    return gpu_futures


