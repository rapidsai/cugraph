# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

# This file test the Renumbering features

import gc
import pytest

import pandas
import numpy as np

import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import cugraph
import dask_cudf
import dask
import cudf
from dask_cuda import LocalCUDACluster
from cugraph.tests import utils
from cugraph.structure.number_map import NumberMap
from cugraph.dask.common.mg_utils import is_single_gpu


@pytest.fixture
def client_connection():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize(p2p=True)

    yield client

    Comms.destroy()
    client.close()
    cluster.close()


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED)
def test_mg_renumber(graph_file, client_connection):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf["src_old"] = sources
    gdf["dst_old"] = destinations
    gdf["src"] = sources + translate
    gdf["dst"] = destinations + translate

    ddf = dask.dataframe.from_pandas(gdf, npartitions=2)

    numbering = NumberMap()
    numbering.from_dataframe(ddf, ["src", "src_old"], ["dst", "dst_old"])
    renumbered_df = numbering.add_internal_vertex_id(
        numbering.add_internal_vertex_id(ddf, "src_id", ["src", "src_old"]),
        "dst_id",
        ["dst", "dst_old"],
    )

    check_src = numbering.from_internal_vertex_id(
        renumbered_df, "src_id"
    ).compute()
    check_dst = numbering.from_internal_vertex_id(
        renumbered_df, "dst_id"
    ).compute()

    assert check_src["0"].to_pandas().equals(check_src["src"].to_pandas())
    assert check_src["1"].to_pandas().equals(check_src["src_old"].to_pandas())
    assert check_dst["0"].to_pandas().equals(check_dst["dst"].to_pandas())
    assert check_dst["1"].to_pandas().equals(check_dst["dst_old"].to_pandas())


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED)
def test_mg_renumber2(graph_file, client_connection):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf["src_old"] = sources
    gdf["dst_old"] = destinations
    gdf["src"] = sources + translate
    gdf["dst"] = destinations + translate
    gdf["weight"] = gdf.index.astype(np.float)

    ddf = dask.dataframe.from_pandas(gdf, npartitions=2)

    ren2, num2 = NumberMap.renumber(
        ddf, ["src", "src_old"], ["dst", "dst_old"]
    )

    check_src = num2.from_internal_vertex_id(ren2, "src").compute()
    check_src = check_src.sort_values("weight").reset_index(drop=True)
    check_dst = num2.from_internal_vertex_id(ren2, "dst").compute()
    check_dst = check_dst.sort_values("weight").reset_index(drop=True)

    assert check_src["0"].to_pandas().equals(gdf["src"].to_pandas())
    assert check_src["1"].to_pandas().equals(gdf["src_old"].to_pandas())
    assert check_dst["0"].to_pandas().equals(gdf["dst"].to_pandas())
    assert check_dst["1"].to_pandas().equals(gdf["dst_old"].to_pandas())


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED)
def test_mg_renumber3(graph_file, client_connection):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf["src_old"] = sources
    gdf["dst_old"] = destinations
    gdf["src"] = sources + translate
    gdf["dst"] = destinations + translate
    gdf["weight"] = gdf.index.astype(np.float)

    ddf = dask.dataframe.from_pandas(gdf, npartitions=2)

    ren2, num2 = NumberMap.renumber(
        ddf, ["src", "src_old"], ["dst", "dst_old"]
    )

    test_df = gdf[["src", "src_old"]].head()

    #
    #  This call raises an exception in branch-0.15
    #  prior to this PR
    #
    test_df = num2.add_internal_vertex_id(test_df, "src", ["src", "src_old"])
    assert True


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_pagerank(client_connection):
    gc.collect()

    pandas.set_option("display.max_rows", 10000)

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

    # Pre compute local data
    # dg.compute_local_data(by='dst')

    expected_pr = cugraph.pagerank(g)
    result_pr = dcg.pagerank(dg).compute()

    err = 0
    tol = 1.0e-05

    assert len(expected_pr) == len(result_pr)

    compare_pr = expected_pr.merge(
        result_pr, on="vertex", suffixes=["_local", "_dask"]
    )

    for i in range(len(compare_pr)):
        diff = abs(
            compare_pr["pagerank_local"].iloc[i]
            - compare_pr["pagerank_dask"].iloc[i]
        )
        if diff > tol * 1.1:
            err = err + 1
    print("Mismatches:", err)
    assert err == 0
