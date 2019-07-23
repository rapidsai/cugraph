import operator
from dask.distributed import Client, wait, default_client, futures_of
from dask_cuda import LocalCUDACluster
import gc
from itertools import product
import time
import numpy as np

import pytest
# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

def test_pagerank():
    gc.collect()
    input_data_path = r"datasets/hibench_small/1/part-00000.csv"

    # Cugraph snmg pagerank Call
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)
    import dask_cudf
    import dask_cugraph.pagerank as dcg
    
    t0 = time.time()
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(input_data_path, chunksize = chunksize, delimiter='\t', names=['src', 'dst'], dtype=['int32', 'int32'])
    y = ddf.to_delayed()
    x = client.compute(y)
    wait(x)
    print (ddf)
    ddf = client.submit(ddf.drop_duplicates())
    print (ddf)
    t1 = time.time()
    print("Reading Csv time: ", t1-t0)
    pr = dcg.pagerank(x, alpha=0.85, max_iter=50)
    t2 = time.time()
    print("Running PR algo time: ", t2-t1)
    res_df = pr.compute()
