import pytest
from cugraph.structure.new_number_map import NumberMap

def test_renumber():

    from dask_cuda import LocalCUDACluster;from dask.distributed import Client;from cugraph.comms import comms as Comms
    cluster = LocalCUDACluster();client = Client(cluster);
    import cudf;df=cudf.DataFrame();df['src']=cudf.Series([1,2,3,4,5,5,4,3,1,0], dtype ='int32');df['dst']=cudf.Series([0,1,2,3,4,1,0,2,3,3], dtype='int32')
    Comms.initialize(p2p=True)
    import dask_cudf as dc;ddf=dc.from_cudf(df, npartitions=2)
    NumberMap.renumber(ddf, 'src','dst')
