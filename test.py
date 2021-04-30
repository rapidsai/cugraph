import cugraph
import cudf
import random
from cugraph.tests import utils
import pandas as pd
import dask.dataframe as dd
df = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1., 2., 3., 4., 5.]})
ddf = dd.from_pandas(df, npartitions=2)

def myadd(df, a, b=1):
    return df.x + df.y + a + b
res = ddf.map_partitions(len).compute()

print(res)

