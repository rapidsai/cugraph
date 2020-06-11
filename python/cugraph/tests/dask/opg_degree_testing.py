from dask.distributed import Client
import gc
import cudf

import cugraph
import dask_cudf

# Move to conftest
from dask_cuda import LocalCUDACluster


# MOVE TO UTILS
def get_n_gpus():
    import os
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


def test_dask_opg_degree():

    gc.collect()
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    client = Client(cluster)

    input_data_path = r"../datasets/karate.csv"

    chunksize = get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    df = cudf.read_csv(input_data_path,
                       delimiter=' ',
                       names=['src', 'dst', 'value'],
                       dtype=['int32', 'int32', 'float32'])

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf)

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst')

    assert dg.in_degree().equals(g.in_degree())
    client.close()
    cluster.close()
