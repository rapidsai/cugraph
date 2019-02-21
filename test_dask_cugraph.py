##TESTING DASK_CUGRAPH
import logging
#from tornado import gen
from dask.distributed import Client, wait, default_client, futures_of
#from dask_cuda import LocalCUDACluster

#cluster = LocalCUDACluster(threads_per_worker=10)
#client = Client(cluster)
client = Client(scheduler_file = "/home/iroy/dask_cugraph/cluster.json", direct_to_workers = True)
print(dir(client.cluster))
#client.close()
devs = [0, 1, 2, 3]
workers = list(client.has_what().keys())
worker_devs = workers[0:min(len(devs), len(workers))]
print("Worker_devs ",worker_devs) 
def set_visible(i, n):
    import os
    all_devices = list(range(n))
    vd = ",".join(map(str, all_devices[i:] + all_devices[:i]))
    print("vd: ", vd)
    os.environ["CUDA_VISIBLE_DEVICES"] = vd
    
[client.submit(set_visible, dev, len(devs), workers = [worker]) for dev, worker in zip(devs, worker_devs)]


import dask.dataframe as dd
import dask_cudf
from toolz import first, assoc
import cudf
import numba.cuda as cuda
import numpy as np

import pandas.testing

from collections import defaultdict

import dask_cugraph as dcg


#input_df = cudf.DataFrame({'a':[1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0],'b':[0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,6,6,6,6,7,7,7,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,10,10,11,11,11,12,12,13,13,13,14,14,14,15,15,15,16,16,16,17,17,17,17,17,18,18,18,19,19,19,20,20,20,21,21,22,22,22,22,23,23,23,24,24,24,24,24,25,25,25,25,26,26,26,26,27,27,28,28,28,28,28,29,29,29,29,30,30,30,31,31,31,31,31]})
input_df = cudf.DataFrame({'src':[0,1,2,3,6,25,0,1,8,20,27,1,2,5,7,8,28,2,3,4,11,2,4,22,26,0,5,15,2,6,13,20,30,0,7,11,16,26,6,8,9,12,18,22,26,0,9,10,20,22,24,26,1,10,14,17,28,5,11,23,10,12,2,13,1,14,19,3,15,21,3,15,16,5,9,17,19,29,0,18,25,7,15,19,2,20,31,10,21,1,16,20,22,11,23,25,5,14,17,23,24,12,17,21,25,4,23,25,26,8,27,2,4,26,28,31,11,16,22,29,12,13,30,23,27,31],'dst':[0,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,6,6,6,6,6,7,7,7,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,9,10,10,10,10,10,11,11,11,12,12,13,13,14,14,14,15,15,15,16,16,16,17,17,17,17,17,18,18,18,19,19,19,20,20,20,21,21,22,22,22,22,23,23,23,24,24,24,24,24,25,25,25,25,26,26,26,26,27,27,28,28,28,28,28,29,29,29,29,30,30,30,31,31,31]})

print(input_df)
ddf = dask_cudf.from_cudf(input_df, chunksize=32).persist()
print("DASK CUDF: ", ddf)
print("CALLING DASK MG PAGERANK")

sdf = ddf.sort_values_binned(by='dst')
pr = dcg.mg_pagerank(sdf)

print(pr)

client.close()
