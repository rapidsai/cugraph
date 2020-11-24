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

import cugraph.dask as dcg
from dask.distributed import Client, default_client, futures_of, wait
import gc
import cugraph
import dask_cudf
import cugraph.comms as Comms
from dask_cuda import LocalCUDACluster
import pytest
from cugraph.dask.common.part_utils import concat_within_workers
from cugraph.dask.common.read_utils import get_n_workers
from cugraph.dask.common.mg_utils import is_single_gpu
import os
import time
import numpy as np
from cugraph.tests import utils


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


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
def test_from_edgelist(client_connection):
    input_data_path = r"../datasets/karate.csv"
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg1 = cugraph.from_edgelist(
        ddf, source="src", destination="dst", edge_attr="value",
        create_using=cugraph.DiGraph)

    dg2 = cugraph.DiGraph()
    dg2.from_dask_cudf_edgelist(
        ddf, source="src", destination="dst", edge_attr="value"
    )

    assert dg1.EdgeList == dg2.EdgeList


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_compute_local_data(client_connection):

    input_data_path = r"../datasets/karate.csv"
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(
        ddf, source="src", destination="dst", edge_attr="value"
    )

    # Compute_local_data
    dg.compute_local_data(by="dst")
    data = dg.local_data["data"]
    by = dg.local_data["by"]

    assert by == "dst"
    assert Comms.is_initialized()

    global_num_edges = data.local_data["edges"].sum()
    assert global_num_edges == dg.number_of_edges()
    global_num_verts = data.local_data["verts"].sum()
    assert global_num_verts == dg.number_of_nodes()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.skip(reason="MG not supported on CI")
def test_parquet_concat_within_workers(client_connection):
    if not os.path.exists("test_files_parquet"):
        print("Generate data... ")
        os.mkdir("test_files_parquet")
    for x in range(10):
        if not os.path.exists("test_files_parquet/df" + str(x)):
            df = utils.random_edgelist(
                e=100, ef=16, dtypes={"src": np.int32, "dst": np.int32}, seed=x
            )
            df.to_parquet("test_files_parquet/df" + str(x), index=False)

    n_gpu = get_n_workers()

    print("Read_parquet... ")
    t1 = time.time()
    ddf = dask_cudf.read_parquet(
        "test_files_parquet/*", dtype=["int32", "int32"]
    )
    ddf = ddf.persist()
    futures_of(ddf)
    wait(ddf)
    t1 = time.time() - t1
    print("*** Read Time: ", t1, "s")
    print(ddf)

    assert ddf.npartitions > n_gpu

    print("Drop_duplicates... ")
    t2 = time.time()
    ddf.drop_duplicates(inplace=True)
    ddf = ddf.persist()
    futures_of(ddf)
    wait(ddf)
    t2 = time.time() - t2
    print("*** Drop duplicate time: ", t2, "s")
    assert t2 < t1

    print("Repartition... ")
    t3 = time.time()
    # Notice that ideally we would use :
    # ddf = ddf.repartition(npartitions=n_gpu)
    # However this is slower than reading and requires more memory
    # Using custom concat instead
    client = default_client()
    ddf = concat_within_workers(client, ddf)
    ddf = ddf.persist()
    futures_of(ddf)
    wait(ddf)
    t3 = time.time() - t3
    print("*** repartition Time: ", t3, "s")
    print(ddf)

    assert t3 < t1
