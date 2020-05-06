import cugraph.dask.opg_pagerank as dcg
from dask.distributed import Client
import gc
import cudf

import cugraph
import dask_cudf

## Move to conftest
from dask_cuda import LocalCUDACluster
#cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
##

def test_dask_pagerank():

    gc.collect()
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    client = Client(cluster)

    input_data_path = r"../datasets/karate.csv"

    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])



    g = cugraph.DiGraph()
    g.from_dask_cudf_edgelist(ddf)

    dcg.pagerank(g)

    client.close()
    cluster.close()
