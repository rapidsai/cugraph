# %%
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
import cugraph.dask.comms as Comms
import cugraph.dask as dask_cugraph

input_data_path = '../datasets/karate.csv'

cluster = LocalCUDACluster()
client = Client(cluster)
Comms.initialize(p2p=True)

# Helper function to set the reader chunk size to automatically get one partition per GPU
chunksize = dask_cugraph.get_chunksize(input_data_path)

# Multi-GPU CSV reader
e_list = dask_cudf.read_csv(input_data_path,
        chunksize = chunksize,
        delimiter=' ',
        names=['src', 'dst'],
        dtype=['int32', 'int32'])

G = cugraph.DiGraph()
G.from_dask_cudf_edgelist(e_list, source='src', destination='dst')

# now run PageRank
pr_df = dask_cugraph.pagerank(G, tol=1e-4)

# All done, clean up
Comms.destroy()
client.close()
cluster.close()

# %%



