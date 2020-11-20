# Copyright (c) 2020, NVIDIA CORPORATION.
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

# import numpy as np
import pytest
import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import gc
import cugraph
import dask_cudf
import cudf
from dask_cuda import LocalCUDACluster
from cugraph.dask.common.mg_utils import is_single_gpu

# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values


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
def test_dask_katz_centrality(client_connection):
    gc.collect()

    input_data_path = r"../datasets/karate.csv"
    chunksize = dcg.get_chunksize(input_data_path)

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

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, "src", "dst")

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    largest_out_degree = g.degrees().nlargest(n=1, columns="out_degree")
    largest_out_degree = largest_out_degree["out_degree"].iloc[0]
    katz_alpha = 1 / (largest_out_degree + 1)

    mg_res = dcg.katz_centrality(dg, alpha=katz_alpha, tol=1e-6)
    mg_res = mg_res.compute()

    import networkx as nx
    from cugraph.tests import utils
    NM = utils.read_csv_for_nx(input_data_path)
    Gnx = nx.from_pandas_edgelist(
        NM, create_using=nx.DiGraph(), source="0", target="1"
    )
    nk = nx.katz_centrality(Gnx, alpha=katz_alpha)
    import pandas as pd
    pdf = pd.DataFrame(nk.items(), columns=['vertex', 'katz_centrality'])
    exp_res = cudf.DataFrame(pdf)
    err = 0
    tol = 1.0e-05

    compare_res = exp_res.merge(
        mg_res, on="vertex", suffixes=["_local", "_dask"]
    )

    for i in range(len(compare_res)):
        diff = abs(
            compare_res["katz_centrality_local"].iloc[i]
            - compare_res["katz_centrality_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0
