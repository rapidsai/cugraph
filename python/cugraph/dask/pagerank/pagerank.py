import cugraph
import random
import dask_cudf as dc
from collections import defaultdict
from cugraph.dask.core import new_ipc_thread, parse_host_port
from cugraph.dask.core import device_of_devicendarray, get_device_id
import os
from dask.distributed import wait, default_client
from toolz import first
import dask.dataframe as dd
import cudf


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
    return (gpu_array[0].__cuda_array_interface__,
            gpu_array[1].__cuda_array_interface__)


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
    # TO DO: IMPROVE/ CLEANUP
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

    pr = cugraph.mg_pagerank(final_allocs_src,
                             final_allocs_dest,
                             alpha, max_iter)

    [t[1].close() for t in open_ipcs]
    [t[1].join() for t in open_ipcs]

    return pr


def pagerank(edge_list, alpha=0.85, max_iter=30):
    """
    Find the PageRank values for each vertex in a graph using multiple GPUs.
    cuGraph computes an approximation of the Pagerank using the power method.
    The input edge list should be provided in dask-cudf dataframe
    with one partition per GPU.

    Parameters
    ----------
    edge_list : dask_cudf.DataFrame
        Contain the connectivity information as an edge list.
        Source 'src' and destination 'dst' columns must be of type 'int32'.
        Edge weights are not used for this algorithm.
        Indices must be in the range [0, V-1], where V is the global number
        of vertices.
    alpha : float
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.
    max_iter : int
        The maximum number of iterations before an answer is returned.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 30.

    Returns
    -------
    PageRank : dask_cudf.DataFrame
        Dask GPU DataFrame containing two columns of size V: the vertex
        identifiers and the corresponding PageRank values.

    Examples
    --------
    >>> import dask_cugraph.pagerank as dcg
    >>> chunksize = dcg.get_chunksize(edge_list.csv)
    >>> ddf_edge_list = dask_cudf.read_csv(edge_list.csv,
    >>>                                    chunksize = chunksize,
    >>>                                    delimiter='\t',
    >>>                                    names=['src', 'dst'],
    >>>                                    dtype=['int32', 'int32'])
    >>> pr = dcg.pagerank(ddf_edge_list, alpha=0.85, max_iter=50)
    """

    client = default_client()
    gpu_futures = _get_mg_info(edge_list)
    # npartitions = len(gpu_futures)

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
    pr = [client.submit(_mg_pagerank,
                        (ipc_handles, raw_arrays, alpha, max_iter),
                        workers=[exec_node])]
    c = cudf.DataFrame({'vertex': cudf.Series(dtype='int32'),
                       'pagerank': cudf.Series(dtype='float32')})
    ddf = dc.from_delayed(pr, meta=c)
    return ddf


def _get_mg_info(ddf):
    # Get gpu data pointers of columns of each dataframe partition

    client = default_client()

    if isinstance(ddf, dd.DataFrame):
        parts = ddf.to_delayed()
        parts = client.compute(parts)
        wait(parts)
    else:
        parts = ddf
    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = client.who_has(parts)
    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = [(worker, client.submit(to_gpu_array, part, workers=[worker]))
                for worker, part in worker_map]

    wait(gpu_data)
    return gpu_data


# UTILITY FUNCTIONS


def _drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    return df


def drop_duplicates(ddf):
    client = default_client()

    if isinstance(ddf, dd.DataFrame):
        parts = ddf.to_delayed()
        parts = client.compute(parts)
        wait(parts)
    else:
        parts = ddf
    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = client.who_has(parts)
    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = [client.submit(_drop_duplicates, part, workers=[worker])
                for worker, part in worker_map]

    wait(gpu_data)
    return gpu_data


def get_n_gpus():
    try:
        return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    except KeyError:
        return len(os.popen("nvidia-smi -L").read().strip().split("\n"))


def get_chunksize(input_path):
    """
    Calculate the appropriate chunksize for dask_cudf.read_csv
    to get a number of partitions equal to the number of GPUs

    Examples
    --------
    >>> import dask_cugraph.pagerank as dcg
    >>> chunksize = dcg.get_chunksize(edge_list.csv)
    """

    import os
    from glob import glob
    import math

    input_files = sorted(glob(str(input_path)))
    if len(input_files) == 1:
        size = os.path.getsize(input_files[0])
        chunksize = math.ceil(size/get_n_gpus())
    else:
        size = [os.path.getsize(_file) for _file in input_files]
        chunksize = max(size)
    return chunksize


def _read_csv(input_files, delimiter, names, dtype):
    df = []
    for f in input_files:
        df.append(cudf.read_csv(f, delimiter=delimiter, names=names,
                                dtype=dtype))
    df_concatenated = cudf.concat(df)
    return df_concatenated


def read_split_csv(input_files, delimiter='\t', names=['src', 'dst'],
                   dtype=['int32', 'int32']):
    """
    Read csv for large datasets which cannot be read directly by dask-cudf
    read_csv due to memory requirements. This function takes large input
    split into smaller files (number of input_files > number of gpus),
    reads two or more csv per gpu/worker and concatenates them into a
    single dataframe. Additional parameters (delimiter, names and dtype)
    can be specified for reading the csv file.
    """

    client = default_client()
    n_files = len(input_files)
    n_gpus = get_n_gpus()
    n_files_per_gpu = int(n_files/n_gpus)
    worker_map = []
    for i, w in enumerate(client.has_what().keys()):
        files_per_gpu = input_files[i*n_files_per_gpu: (i+1)*n_files_per_gpu]
        worker_map.append((files_per_gpu, w))
    new_ddf = [client.submit(_read_csv, part, delimiter, names, dtype,
               workers=[worker]) for part, worker in worker_map]

    wait(new_ddf)
    return new_ddf
