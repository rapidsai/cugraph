import warnings
import gc
import time

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


def test_splitting():
    gc.collect()

    # This is an experimental setup for 300GB bigdatax8 dataset.
    # This test can be run on 16 32GB gpus. The dataset is split into 32 files.
    input_data_path = r"/datasets/pagerank_demo/1/Input-bigdatax8/edges/"
    input_files = ['file-00000.csv',
                   'file-00001.csv',
                   'file-00002.csv',
                   'file-00003.csv',
                   'file-00004.csv',
                   'file-00005.csv',
                   'file-00006.csv',
                   'file-00007.csv',
                   'file-00008.csv',
                   'file-00009.csv',
                   'file-00010.csv',
                   'file-00011.csv',
                   'file-00012.csv',
                   'file-00013.csv',
                   'file-00014.csv',
                   'file-00015.csv',
                   'file-00016.csv',
                   'file-00017.csv',
                   'file-00018.csv',
                   'file-00019.csv',
                   'file-00020.csv',
                   'file-00021.csv',
                   'file-00022.csv',
                   'file-00023.csv',
                   'file-00024.csv',
                   'file-00025.csv',
                   'file-00026.csv',
                   'file-00027.csv',
                   'file-00028.csv',
                   'file-00029.csv',
                   'file-00030.csv',
                   'file-00031.csv']

    # Cugraph snmg pagerank Call
    cluster = LocalCUDACluster(threads_per_worker=1)
    client = Client(cluster)

    files = [input_data_path+f for f in input_files]

    # Read 2 files per gpu/worker and concatenate the dataframe
    # This is a work around for large files to fit memory requirements
    # of cudf.read_csv
    t0 = time.time()
    new_ddf = dcg.read_split_csv(files)
    t1 = time.time()
    print("Reading Csv time: ", t1-t0)
    t2 = time.time()
    pr = dcg.pagerank(new_ddf, alpha=0.85, max_iter=3)
    wait(pr)
    t3 = time.time()
    print("Pagerank (Dask) time: ", t3-t2)
    t4 = time.time()
    res_df = pr.compute()
    t5 = time.time()
    print("Compute time: ", t5-t4)
    print(res_df)
    t6 = time.time()
    res_df.to_csv('~/pagerank.csv', chunksize=40000000, header=False,
                  index=False)
    t7 = time.time()
    print("Write csv time: ", t7-t6)

    client.close()
    cluster.close()
