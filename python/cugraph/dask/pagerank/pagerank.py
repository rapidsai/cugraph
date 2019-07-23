import cugraph 
import cudf
import numpy as np
import random 
import dask.dataframe as dd 
import dask_cudf as dc
from collections import defaultdict
import numba.cuda
from dask_cugraph.core import new_ipc_thread, parse_host_port
from dask_cugraph.core import device_of_devicendarray, get_device_id
import os
from dask.distributed import Client, wait, default_client, futures_of
from toolz import first, assoc


def to_gpu_array(df):
    """
    Get the gpu_array pointer to the data in columns of the
    input dataframe.
    """
    start_idx = df.index[0]
    stop_idx = df.index[-1]
    gpu_array_src = df['src']._column._data.mem
    gpu_array_dest = df['dst']._column._data.mem
    dev = device_of_devicendarray(gpu_array_src)
    return dev, (gpu_array_src, gpu_array_dest), (start_idx, stop_idx)


def build_alloc_info(data):
    """
    Use the __cuda_array_interface__ to extract cpointer
    information for passing into cython.
    """
    dev, gpu_array, _ = data
    return (gpu_array[0].__cuda_array_interface__, gpu_array[1].__cuda_array_interface__)


def get_ipc_handle(data):
    """
    Extract IPC handles from input Numba array. Pass
    along the device of the current worker and the
    start/stop indices from the original cudf.
    """
    dev, gpu_array, idx = data

    in_handle_src = gpu_array[0].get_ipc_handle()
    in_handle_dest = gpu_array[1].get_ipc_handle()
    return dev, [in_handle_src, in_handle_dest], idx


def _build_host_dict(gpu_futures, client):
    """ 
    Build a dictionary of hosts and their corresponding ports from workers 
    which have the given gpu_futures.
    """
    ######## TO DO: IMPROVE/ CLEANUP
    who_has = client.who_has(gpu_futures)

    workers = [key[0] for key in list(who_has.values())]
    hosts = set(map(lambda x: parse_host_port(x), workers))
    hosts_dict = {}
    for host, port in hosts:
        if host not in hosts_dict:
            hosts_dict[host] = set([port])
        else:
            hosts_dict[host].add(port)

    return hosts_dict


'''def get_dev_idx(data):
    ipcs, raw_arrs = data
    device_handle_map = defaultdict(list)
    [device_handle_map[dev].append((idx, ipc)) for dev, ipc, idx in ipcs]

    dev_idxs = []
    for dev, ipcs in device_handle_map.items():
        dev_idxs.append(get_device_id(dev))
    return [dev_idxs, os.environ['CUDA_VISIBLE_DEVICES']]
'''

def _mg_pagerank(data):
    """
    Collect all ipc pointer information into source and destination alloc_info 
    list that is passed to snmg pagerank.
    """ 
    ipcs, raw_arrs, alpha, max_iter = data
    
    # Separate threads to hold pointers to separate devices
    # The order in which we pass the list of IPCs to the thread matters and
    # the goal isto maximize reuse while minimizing the number of threads.
    # We want to limit the number of threads to O(len(devices)) and want to
    # avoid having if be O(len(ipcs)) at all costs!
    device_handle_map = defaultdict(list)
    [device_handle_map[dev].append((idx, ipc)) for dev, ipc, idx in ipcs]
    
    '''open_ipcs = [([i[0] for i in ipcs],
                  new_ipc_thread([i[1] for i in ipcs], dev))
                 for dev, ipcs in device_handle_map.items()]
    '''
    # TO DO: Improve/Generalize the calculation of open_ipcs/alloc_info 
    '''open_ipcs = []
    for dev, ipcs in device_handle_map.items():
        open_ipcs.append([[ipcs[0][0]], new_ipc_thread(ipcs[0][1], dev)])

    alloc_info_src = []
    alloc_info_dest = []
    for idxs, t in open_ipcs:
        inf = t.info()
        for i in range(len(idxs)):
            alloc_info_src.append([idxs[i], inf[0]])
            alloc_info_dest.append([idxs[i], inf[1]])

    for t in raw_arrs:
        raw_info = build_alloc_info(t)
        alloc_info_src.append([t[2], raw_info[0]])
        alloc_info_dest.append([t[2], raw_info[1]])

    alloc_info_src.sort(key=lambda x: x[0][0])
    alloc_info_dest.sort(key=lambda x: x[0][0])
    final_allocs_src = [a for i, a in alloc_info_src]
    final_allocs_dest = [a for i, a in alloc_info_dest]
    '''
    open_ipcs = []
    for dev, ipcs in device_handle_map.items():
        open_ipcs.append([[dev], new_ipc_thread(ipcs[0][1], dev)])

    alloc_info_src = []
    alloc_info_dest = []
    for dev, t in open_ipcs:
        inf = t.info()
        for i in range(len(dev)):
            alloc_info_src.append([get_device_id(dev[i]), inf[0]])
            alloc_info_dest.append([get_device_id(dev[i]), inf[1]])

    for t in raw_arrs:
        raw_info = build_alloc_info(t)
        alloc_info_src.append([get_device_id(t[2]), raw_info[0]])
        alloc_info_dest.append([get_device_id(t[2]), raw_info[1]])

    alloc_info_src.sort(key=lambda x: x[0])
    alloc_info_dest.sort(key=lambda x: x[0])

    final_allocs_src = [a for i, a in alloc_info_src]
    final_allocs_dest = [a for i, a in alloc_info_dest]
    
    '''ipc_dev_list, devarrs_dev_list = data

    # Open 1 ipc thread per device
    open_ipcs = []
    print(ipc_dev_list)
    for dev, p, idx in ipc_dev_list:
        print("P: ",p)
        arrs = []
        for x, y in p:
            arrs.append(x)
            arrs.append(y)
            #arrs.append(coef)
        ipct = new_ipc_thread(arrs, dev)
        open_ipcs.append(ipct)

    print("OPEN IPCS: ",open_ipcs)
    alloc_info = []
    for t in open_ipcs:
        outsiders = t.info()
        triplet = []
        for i in range(0, len(outsiders), 3):
            triplet.append(outsiders[i])
            triplet.append(outsiders[i+1])
            triplet.append(outsiders[i+2])
            alloc_info.append(triplet)
    
    for dev, p, idx in devarrs_dev_list:
        locals = []
        for X, coef, pred in p:
            locals.append(build_alloc_info(X)[0])
            locals.append(build_alloc_info(coef)[0])
            locals.append(build_alloc_info(pred)[0])
        alloc_info.append(locals)
    '''
    pr = cugraph.mg_pagerank(final_allocs_src, final_allocs_dest, alpha, max_iter)
    
    [t[1].close() for t in open_ipcs]
    [t[1].join() for t in open_ipcs]
    
    return pr


def pagerank(ddf, alpha=0.85, max_iter=30):
    client = default_client()
    gpu_futures = _get_mg_info(ddf)
    npartitions = len(gpu_futures)

    host_dict = _build_host_dict(gpu_futures, client).items()
    if len(host_dict) > 1:
        raise Exception("Dask cluster appears to span hosts. Current "
                        "multi-GPU version is limited to single host")

    master_host = [(host, random.sample(ports, 1)[0])
                        for host, ports in host_dict][0]

    host, port = master_host
    gpu_futures_for_host = list(filter(lambda d: d[0][0] == host,
                                           gpu_futures))
    exec_node = (host, port)
    # build ipc handles
    gpu_data_excl_worker = list(filter(lambda d: d[0] != exec_node,
                                       gpu_futures_for_host))
    gpu_data_incl_worker = list(filter(lambda d: d[0] == exec_node,
                                       gpu_futures_for_host))

    ipc_handles = [client.submit(get_ipc_handle, future, workers=[worker])
                   for worker, future in gpu_data_excl_worker]

    raw_arrays = [future for worker, future in gpu_data_incl_worker]

    '''dev_idx = client.submit(get_dev_idx, (ipc_handles, raw_arrays),
                                  workers=[exec_node]).result()
    print("dev_idx: ",dev_idx)
    '''
    pr = client.submit(_mg_pagerank, (ipc_handles, raw_arrays, alpha, max_iter),
                       workers=[exec_node]).result()

    #wait(pr)
    ddf = dc.from_cudf(pr, npartitions =npartitions)    
    return ddf


def find_dev(df):
    gpu_array_src = df['src']._column._data.mem
    dev = device_of_devicendarray(gpu_array_src)
    return dev

 
def _get_mg_info(ddf):
        # Get gpu data pointers of columns of each dataframe partition

        client = default_client()

        '''if isinstance(ddf, dd.DataFrame):
            parts = ddf.to_delayed()
            parts = client.compute(parts)
            wait(parts)
        '''
        parts = ddf
        key_to_part_dict = dict([(str(part.key), part) for part in parts])
        who_has = client.who_has(parts)
        worker_map = []
        for key, workers in who_has.items():
            worker = parse_host_port(first(workers))
            worker_map.append((worker, key_to_part_dict[key]))

        gpu_data = [(worker, client.submit(to_gpu_array, part,
                                           workers=[worker]))
                    for worker, part in worker_map]

        wait(gpu_data)
        x = [client.submit(find_dev, part, workers=[worker]).result() for worker, part in worker_map]
        return gpu_data

def get_n_gpus():
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        print("here")
        return len(os.popen("nvidia-smi -L").read().strip().split("\n"))


def get_chunksize(input_path, delimiter = ',', names = None, dtype = None):
    # Calculate appropriate chunksize to get partitions equal to number of gpus
    import os
    from glob import glob
    import math
    #from numba import cuda
    import dask_cudf
    input_files = sorted(glob(str(input_path)))
    if len(input_files) is 1:
        size = os.path.getsize(input_files[0])
        chunksize = math.ceil(size/get_n_gpus())
    else:
        size = [os.path.getsize(_file) for _file in input_files]
        chunksize = max(size)
    return chunksize    