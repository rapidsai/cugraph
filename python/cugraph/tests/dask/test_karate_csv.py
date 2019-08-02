import warnings
import gc
import dask_cudf
import pandas as pd
# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from dask.distributed import Client
    import cugraph.dask.pagerank as dcg
    from dask_cuda import LocalCUDACluster
    import networkx as nx


def test_pagerank():
    gc.collect()
    input_data_path = r"../datasets/karate.csv"
    # Networkx Call
    pd_df = pd.read_csv(input_data_path, delimiter=' ',
                        names=['src', 'dst', 'value'])
    G = nx.Graph()
    for i in range(0, len(pd_df)):
        G.add_edge(pd_df['src'][i], pd_df['dst'][i])
    nx_pr = nx.pagerank(G, alpha=0.85)
    nx_pr = sorted(nx_pr.items(), key=lambda x: x[0])
    # Cugraph snmg pagerank Call
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    pr = dcg.pagerank(ddf, alpha=0.85, max_iter=50)
    res_df = pr.compute()

    err = 0
    tol = 1.0e-05
    for i in range(len(res_df)):
        if(abs(res_df['pagerank'][i]-nx_pr[i][1]) > tol*1.1):
            err = err + 1
    print("Mismatches:", err)
    assert err < (0.01*len(res_df))

    client.close()
    cluster.close()
