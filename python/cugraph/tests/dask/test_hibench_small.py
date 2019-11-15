import warnings
import gc
import dask_cudf
import pandas as pd
import time
import tempfile
import os

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from dask.distributed import Client, wait
    import cugraph.dask.pagerank as dcg
    from dask_cuda import LocalCUDACluster
    import networkx as nx


def test_pagerank():
    gc.collect()
    input_data_path = r"../datasets/hibench_small/1/part-00000.csv"

    # Networkx Call
    pd_df = pd.read_csv(input_data_path, delimiter='\t', names=['src', 'dst'])
    G = nx.DiGraph()
    for i in range(0, len(pd_df)):
        G.add_edge(pd_df['src'][i], pd_df['dst'][i])
    nx_pr = nx.pagerank(G, alpha=0.85)
    nx_pr = sorted(nx_pr.items(), key=lambda x: x[0])

    # Cugraph snmg pagerank Call
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    t0 = time.time()
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter='\t', names=['src', 'dst'],
                             dtype=['int32', 'int32'])
    y = ddf.to_delayed()
    x = client.compute(y)
    wait(x)
    t1 = time.time()
    print("Reading Csv time: ", t1-t0)
    new_ddf = dcg.drop_duplicates(x)
    t2 = time.time()
    pr = dcg.pagerank(new_ddf, alpha=0.85, max_iter=50)
    wait(pr)
    t3 = time.time()
    print("Running PR algo time: ", t3-t2)
    t4 = time.time()
    res_df = pr.compute()
    t5 = time.time()
    print("Compute time: ", t5-t4)
    print(res_df)

    # Use tempfile.mkstemp() to get a temp file name. Close and delete the file
    # so to_csv() can create it using the unique temp name
    (tempfileHandle, tempfileName) = tempfile.mkstemp(suffix=".csv",
                                                      prefix="pagerank_")
    os.close(tempfileHandle)
    os.remove(tempfileName)

    # For bigdatax4, chunksize=100000000 to avoid oom on write csv
    t6 = time.time()
    res_df.to_csv(tempfileName, header=False, index=False)
    t7 = time.time()
    print("Write csv time: ", t7-t6)

    # Comparison
    err = 0
    tol = 1.0e-05
    for i in range(len(res_df)):
        if(abs(res_df['pagerank'][i]-nx_pr[i][1]) > tol*1.1):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.02*len(res_df))

    client.close()
    cluster.close()
    os.remove(tempfileName)
