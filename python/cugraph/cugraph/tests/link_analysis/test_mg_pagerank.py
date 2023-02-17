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
import numpy as np
import pytest
import cugraph.dask as dcg
import gc
import cugraph
import dask_cudf
from cugraph.testing import utils
import cudf

# from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH


# The function selects personalization_perc% of accessible vertices in graph M
# and randomly assigns them personalization values


def personalize(vertices, personalization_perc):
    personalization = None
    if personalization_perc != 0:
        personalization = {}
        nnz_vtx = vertices.values_host
        personalization_count = int((nnz_vtx.size * personalization_perc) / 100.0)
        nnz_vtx = np.random.choice(
            nnz_vtx, min(nnz_vtx.size, personalization_count), replace=False
        )
        nnz_val = np.random.random(nnz_vtx.size)
        nnz_val = nnz_val / sum(nnz_val)
        for vtx, val in zip(nnz_vtx, nnz_val):
            personalization[vtx] = val

        k = np.fromiter(personalization.keys(), dtype="int32")
        v = np.fromiter(personalization.values(), dtype="float32")
        cu_personalization = cudf.DataFrame({"vertex": k, "values": v})

    return cu_personalization, personalization


# =============================================================================
# Parameters
# =============================================================================
PERSONALIZATION_PERC = [0, 10, 50]
IS_DIRECTED = [True, False]
HAS_GUESS = [0, 1]
HAS_PRECOMPUTED = [0, 1]


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.mark.parametrize("personalization_perc", PERSONALIZATION_PERC)
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("has_precomputed_vertex_out_weight", HAS_PRECOMPUTED)
@pytest.mark.parametrize("has_guess", HAS_GUESS)
def test_dask_pagerank(
    dask_client,
    personalization_perc,
    directed,
    has_precomputed_vertex_out_weight,
    has_guess,
):

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

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    g = cugraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, "src", "dst", "value")

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "value", store_transposed=True)

    personalization = None
    pre_vtx_o_wgt = None
    nstart = None
    max_iter = 100
    has_precomputed_vertex_out_weight
    if personalization_perc != 0:
        personalization, p = personalize(g.nodes(), personalization_perc)
    if has_precomputed_vertex_out_weight == 1:
        df = df[["src", "value"]]
        pre_vtx_o_wgt = (
            df.groupby(["src"], as_index=False)
            .sum()
            .rename(columns={"src": "vertex", "value": "sums"})
        )

    if has_guess == 1:
        nstart = cugraph.pagerank(g, personalization=personalization, tol=1e-6).rename(
            columns={"pagerank": "values"}
        )
        max_iter = 20

    expected_pr = cugraph.pagerank(
        g,
        personalization=personalization,
        precomputed_vertex_out_weight=pre_vtx_o_wgt,
        max_iter=max_iter,
        tol=1e-6,
        nstart=nstart,
    )
    result_pr = dcg.pagerank(
        dg,
        personalization=personalization,
        precomputed_vertex_out_weight=pre_vtx_o_wgt,
        max_iter=max_iter,
        tol=1e-6,
        nstart=nstart,
    )
    result_pr = result_pr.compute()

    err = 0
    tol = 1.0e-05

    assert len(expected_pr) == len(result_pr)

    compare_pr = expected_pr.merge(result_pr, on="vertex", suffixes=["_local", "_dask"])

    for i in range(len(compare_pr)):
        diff = abs(
            compare_pr["pagerank_local"].iloc[i] - compare_pr["pagerank_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    assert err == 0


def test_pagerank_invalid_personalization_dtype(dask_client):
    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()

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
        ddf,
        source="src",
        destination="dst",
        edge_attr="value",
        renumber=True,
        store_transposed=True,
    )

    personalization_vec = cudf.DataFrame()
    personalization_vec["vertex"] = [17, 26]
    personalization_vec["values"] = [0.5, 0.75]
    warning_msg = (
        "PageRank requires 'personalization' values to match the "
        "graph's 'edge_attr' type. edge_attr type is: "
        "float32 and got 'personalization' values "
        "of type: float64."
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.pagerank(dg, personalization=personalization_vec)


def test_dask_pagerank_transposed_false(dask_client):
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
        "Pagerank expects the 'store_transposed' "
        "flag to be set to 'True' for optimal performance during "
        "the graph creation"
    )

    with pytest.warns(UserWarning, match=warning_msg):
        dcg.pagerank(dg)
