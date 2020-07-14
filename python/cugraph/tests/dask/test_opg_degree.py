from dask.distributed import Client
import gc
import cudf
import cugraph.comms.comms as Comms
import cugraph
import dask_cudf

# Move to conftest
from dask_cuda import LocalCUDACluster


def test_dask_opg_degree():

    gc.collect()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    input_data_path = r"../datasets/karate.csv"

    chunksize = cugraph.dask.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    df = cudf.read_csv(input_data_path,
                       delimiter=' ',
                       names=['src', 'dst', 'value'],
                       dtype=['int32', 'int32', 'float32'])

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf)

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst')

    assert dg.in_degree().equals(g.in_degree())

    Comms.destroy()
    client.close()
    cluster.close()
