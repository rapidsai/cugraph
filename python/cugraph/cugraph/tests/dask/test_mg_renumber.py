# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
import dask_cudf
import dask
import cudf
from cudf.testing import assert_series_equal

import cugraph.dask as dcg
import cugraph
from cugraph.tests import utils
from cugraph.structure.number_map import NumberMap
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.tests.utils import RAPIDS_DATASET_ROOT_DIR_PATH


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED,
                         ids=[f"dataset={d.as_posix()}"
                              for d in utils.DATASETS_UNRENUMBERED])
def test_mg_renumber(graph_file, dask_client):

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf["src_old"] = sources
    gdf["dst_old"] = destinations
    gdf["src"] = sources + translate
    gdf["dst"] = destinations + translate

    ddf = dask.dataframe.from_pandas(
        gdf, npartitions=len(dask_client.scheduler_info()['workers']))

    # preserve_order is not supported for MG
    renumbered_df, renumber_map = NumberMap.renumber(ddf,
                                                     ["src", "src_old"],
                                                     ["dst", "dst_old"],
                                                     preserve_order=False)
    unrenumbered_df = renumber_map.unrenumber(renumbered_df, "src",
                                              preserve_order=False)
    unrenumbered_df = renumber_map.unrenumber(unrenumbered_df, "dst",
                                              preserve_order=False)

    # sort needed only for comparisons, since preserve_order is False
    gdf = gdf.sort_values(by=["src", "src_old", "dst", "dst_old"])
    gdf = gdf.reset_index()
    unrenumbered_df = unrenumbered_df.compute()
    unrenumbered_df = unrenumbered_df.sort_values(by=["0_src", "1_src",
                                                      "0_dst", "1_dst"])
    unrenumbered_df = unrenumbered_df.reset_index()

    assert_series_equal(gdf["src"], unrenumbered_df["0_src"],
                        check_names=False)
    assert_series_equal(gdf["src_old"], unrenumbered_df["1_src"],
                        check_names=False)
    assert_series_equal(gdf["dst"], unrenumbered_df["0_dst"],
                        check_names=False)
    assert_series_equal(gdf["dst_old"], unrenumbered_df["1_dst"],
                        check_names=False)


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("graph_file", utils.DATASETS_UNRENUMBERED,
                         ids=[f"dataset={d.as_posix()}"
                              for d in utils.DATASETS_UNRENUMBERED])
def test_mg_renumber_add_internal_vertex_id(graph_file, dask_client):
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

    ddf = dask.dataframe.from_pandas(
        gdf, npartitions=len(dask_client.scheduler_info()['workers']))

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
def test_dask_pagerank(dask_client):
    pandas.set_option("display.max_rows", 10000)

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "karate.csv").as_posix()
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

    g = cugraph.Graph(directed=True)
    g.from_cudf_edgelist(df, "src", "dst")

    dg = cugraph.Graph(directed=True)
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


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("renumber", [False])
def test_directed_graph_renumber_false(renumber, dask_client):
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "karate.csv").as_posix()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    dg = cugraph.Graph(directed=True)

    with pytest.raises(ValueError):
        dg.from_dask_cudf_edgelist(ddf, "src", "dst", renumber=renumber)


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
@pytest.mark.parametrize("renumber", [False])
def test_multi_directed_graph_renumber_false(renumber, dask_client):
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "karate_multi_edge.csv").as_posix()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    dg = cugraph.MultiGraph(directed=True)

    with pytest.raises(ValueError):
        dg.from_dask_cudf_edgelist(ddf, "src", "dst", renumber=renumber)
