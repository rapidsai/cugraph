from dask.distributed import Client
import gc
import pytest
import cudf
import cugraph.comms as Comms
import cugraph
import dask_cudf

# Move to conftest
from dask_cuda import LocalCUDACluster


@pytest.fixture
def client_connection():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    yield client

    Comms.destroy()
    client.close()
    cluster.close()


def test_dask_opg_degree(client_connection):
    gc.collect()

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
    dg.from_dask_cudf_edgelist(ddf, renumber=False)

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst')

    merge_df = dg.in_degree().merge(
        g.in_degree(), on="vertex", suffixes=['_dg', '_g'])

    assert merge_df['degree_dg'].equals(merge_df['degree_g'])
