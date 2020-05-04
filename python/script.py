from cugraph.dask.opg_pagerank import pagerank
from dask.distributed import Client
import gc
import cudf

class gGraph():
    #edgelist=None
    def __init__(self):
        self.edgelist=None
        print("init graph")

import dask_cudf

## Move to conftest
from dask_cuda import LocalCUDACluster
#cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
##

def test_dask_pagerank():

    gc.collect()
    cluster = LocalCUDACluster(protocol="tcp", scheduler_port=0)
    client = Client(cluster)
    print(dir(client))
    print(dir(cluster))
    ## Create temp ddf
    df = cudf.DataFrame()
    df['0']=[1,1,1,2,3,4]
    df['1']=[4,4,0,3,1,2]

    ddf = dask_cudf.from_cudf(df, npartitions=2)
    print(ddf)
    ##

    g = gGraph()
    g.edgelist = ddf

    pagerank(g)

    client.close()
    cluster.close()
