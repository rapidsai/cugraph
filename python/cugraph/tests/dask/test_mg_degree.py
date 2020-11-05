# Copyright (c) 2018-2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dask.distributed import Client
import gc
import pytest
import cudf
import cugraph.comms as Comms
import cugraph
import dask_cudf
from cugraph.dask.common.mg_utils import is_single_gpu

# Move to conftest
from dask_cuda import LocalCUDACluster


@pytest.fixture
def client_connection():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize(p2p=True)

    yield client

    Comms.destroy()
    client.close()
    cluster.close()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_mg_degree(client_connection):
    gc.collect()

    input_data_path = r"../datasets/karate.csv"

    chunksize = cugraph.dask.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, "src", "dst")

    merge_df = (
        dg.in_degree()
        .merge(g.in_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    assert merge_df["degree_dg"].equals(merge_df["degree_g"])
