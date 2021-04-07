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
import cugraph
import dask_cudf
import dask
import cudf
from cugraph.tests import utils
from cugraph.structure.number_map import NumberMap
from cugraph.dask.common.mg_utils import (is_single_gpu,
                                          setup_local_dask_cluster,
                                          teardown_local_dask_cluster)


@pytest.fixture(scope="module")
def client_connection():
    (cluster, client) = setup_local_dask_cluster(p2p=True)
    yield client
    teardown_local_dask_cluster(cluster, client)


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED,
                         ids=[f"dataset={d.as_posix()}"
                              for d in utils.DATASETS_UNRENUMBERED])
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

    renumbered_gdf, renumber_map = NumberMap.renumber(ddf,
                                                      ["src", "src_old"],
                                                      ["dst", "dst_old"],
                                                      preserve_order=True)
    unrenumbered_df = renumber_map.unrenumber(renumbered_df, "src",
                                              preserve_order=True)
    unrenumbered_df = renumber_map.unrenumber(unrenumbered_df, "dst",
                                              preserve_order=True)

    assert gdf["src"].to_pandas().equals(unrenumbered_df["0_src"].to_pandas())
    assert gdf["src_old"].to_pandas().equals(unrenumbered_df["1_src"].to_pandas())
    assert gdf["dst"].to_pandas().equals(unrenumbered_df["0_dst"].to_pandas())
    assert gdf["dst_old"].to_pandas().equals(unrenumbered_df["1_dst"].to_pandas())


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED,
                         ids=[f"dataset={d.as_posix()}"
                              for d in utils.DATASETS_UNRENUMBERED])
def test_mg_renumber_add_internal_vertex_id(graph_file, client_connection):
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
