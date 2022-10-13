# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
import gc
import cugraph
import dask_cudf
import cudf
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_katz_centrality(dask_client, directed):

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(
        ddf, "src", "dst", legacy_renum_only=True, store_transposed=True
    )

    degree_max = dg.degree()["degree"].max().compute()
    katz_alpha = 1 / (degree_max)

    mg_res = dcg.katz_centrality(dg, alpha=katz_alpha, tol=1e-6)
    mg_res = mg_res.compute()

    import networkx as nx
    from cugraph.testing import utils

    NM = utils.read_csv_for_nx(input_data_path)
    if directed:
        Gnx = nx.from_pandas_edgelist(
            NM, create_using=nx.DiGraph(), source="0", target="1"
        )
    else:
        Gnx = nx.from_pandas_edgelist(
            NM, create_using=nx.Graph(), source="0", target="1"
        )
    nk = nx.katz_centrality(Gnx, alpha=katz_alpha)
    import pandas as pd

    pdf = pd.DataFrame(nk.items(), columns=["vertex", "katz_centrality"])
    exp_res = cudf.DataFrame(pdf)
    err = 0
    tol = 1.0e-05

    compare_res = exp_res.merge(mg_res, on="vertex", suffixes=["_local", "_dask"])

    for i in range(len(compare_res)):
        diff = abs(
            compare_res["katz_centrality_local"].iloc[i]
            - compare_res["katz_centrality_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_katz_centrality_nstart(dask_client, directed):
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(
        ddf, "src", "dst", legacy_renum_only=True, store_transposed=True
    )

    mg_res = dcg.katz_centrality(dg, max_iter=50, tol=1e-6)
    mg_res = mg_res.compute()

    estimate = mg_res.copy()
    estimate = estimate.rename(
        columns={"vertex": "vertex", "katz_centrality": "values"}
    )
    estimate["values"] = 0.5

    mg_estimate_res = dcg.katz_centrality(dg, nstart=estimate, max_iter=50, tol=1e-6)
    mg_estimate_res = mg_estimate_res.compute()

    err = 0
    tol = 1.0e-05
    compare_res = mg_res.merge(
        mg_estimate_res, on="vertex", suffixes=["_dask", "_nstart"]
    )

    for i in range(len(compare_res)):
        diff = abs(
            compare_res["katz_centrality_dask"].iloc[i]
            - compare_res["katz_centrality_nstart"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


def test_dask_katz_centrality_transposed_false(dask_client):
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()

    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(
        ddf, "src", "dst", legacy_renum_only=True, store_transposed=False
    )

    warning_msg = (
        "Katz centrality expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.katz_centrality(dg)
