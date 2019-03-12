##TESTING DASK_CUGRAPH
import logging
#from tornado import gen
from dask.distributed import Client, wait, default_client, futures_of
#from dask_cuda import LocalCUDACluster

#cluster = LocalCUDACluster(threads_per_worker=10)
#client = Client(cluster)
client = Client(scheduler_file = "cluster.json", direct_to_workers = True)
devs = [0, 1, 2, 3, 4, 5, 6, 7]
workers = list(client.has_what().keys())
worker_devs = workers[0:min(len(devs), len(workers))]
print("Worker_devs ",worker_devs) 
import cudf
import numpy as np
def set_visible(i, n):
    import os
    all_devices = list(range(n))
    vd = ",".join(map(str, all_devices[i:] + all_devices[:i]))
    print("vd: ", vd)
    os.environ["CUDA_VISIBLE_DEVICES"] = vd
        
ftr = [client.submit(set_visible, dev, len(devs), workers = [worker]) for dev, worker in zip(devs, worker_devs)]
wait(ftr)

import dask.dataframe as dd
import dask_cudf
from toolz import first, assoc
import cudf
import numba.cuda as cuda
import numpy as np

import pandas.testing

from collections import defaultdict

import dask_cugraph as dcg

print("Read Input Data.")
input_data_path = r"/datasets/pagerank_demo/8/Input-gigantic/edges"
dgdf = dask_cudf.read_csv(input_data_path + r"/part-*", chunksize= 3200000000, delimiter='\t', names=['src', 'dst'], dtype=['int32', 'int32']).persist()

print("DASK CUDF: ", dgdf)
print("CALLING DASK MG PAGERANK")

pr = dcg.mg_pagerank(dgdf)
print(pr)
res_df = pr.compute()
print(res_df)
client.close()
