"""
import cugraph
import cudf
import random
from cugraph.tests import utils
import pandas as pd
import dask.dataframe as dd
from dask.distributed import wait, default_client, Client
"""

"""
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1., 2., 3., 4., 5.]})
ddf = dd.from_pandas(df, npartitions=2)

def myadd(df, a, b=1):
    return df.x + df.y + a + b
res = ddf.map_partitions(len).compute()

print(res)
"""

#client = default_client()
"""
if __name__ == '__main__':
    client = Client()
    print(client)
    workers = len(client.scheduler_info()['workers'])
    print(workers)
"""

import cugraph.dask as dcg

chunksize = dcg.get_chunksize("/home/nfs/jnke/learn/datasets/karate.csv")
ddf = dask_cudf.read_csv("/home/nfs/jnke/learn/datasets/karate.csv", chunksize=4,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])
dg = cugraph.DiGraph()
dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
                               edge_attr='value')
pr = dcg.katz_centrality(dg)
