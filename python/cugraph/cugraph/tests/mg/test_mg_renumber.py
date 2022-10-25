# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
from cudf.testing import assert_frame_equal, assert_series_equal

import cugraph.dask as dcg
import cugraph
from cugraph.testing import utils
from cugraph.structure.number_map import NumberMap
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH


# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file",
    utils.DATASETS_UNRENUMBERED,
    ids=[f"dataset={d.as_posix()}" for d in utils.DATASETS_UNRENUMBERED],
)
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
        gdf, npartitions=len(dask_client.scheduler_info()["workers"])
    )

    # preserve_order is not supported for MG
    renumbered_df, renumber_map = NumberMap.renumber(
        ddf, ["src", "src_old"], ["dst", "dst_old"], preserve_order=False
    )
    unrenumbered_df = renumber_map.unrenumber(
        renumbered_df, renumber_map.renumbered_src_col_name, preserve_order=False
    )
    unrenumbered_df = renumber_map.unrenumber(
        unrenumbered_df, renumber_map.renumbered_dst_col_name, preserve_order=False
    )

    # sort needed only for comparisons, since preserve_order is False
    gdf = gdf.sort_values(by=["src", "src_old", "dst", "dst_old"])
    gdf = gdf.reset_index()
    unrenumbered_df = unrenumbered_df.compute()
    src = renumber_map.renumbered_src_col_name
    dst = renumber_map.renumbered_dst_col_name
    unrenumbered_df = unrenumbered_df.sort_values(
        by=[f"0_{src}", f"1_{src}", f"0_{dst}", f"1_{dst}"]
    )
    unrenumbered_df = unrenumbered_df.reset_index()

    assert_series_equal(gdf["src"], unrenumbered_df[f"0_{src}"], check_names=False)
    assert_series_equal(gdf["src_old"], unrenumbered_df[f"1_{src}"], check_names=False)
    assert_series_equal(gdf["dst"], unrenumbered_df[f"0_{dst}"], check_names=False)
    assert_series_equal(gdf["dst_old"], unrenumbered_df[f"1_{dst}"], check_names=False)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file",
    utils.DATASETS_UNRENUMBERED,
    ids=[f"dataset={d.as_posix()}" for d in utils.DATASETS_UNRENUMBERED],
)
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
        gdf, npartitions=len(dask_client.scheduler_info()["workers"])
    )

    ren2, num2 = NumberMap.renumber(ddf, ["src", "src_old"], ["dst", "dst_old"])

    test_df = gdf[["src", "src_old"]].head()

    # simply check that this does not raise an exception
    num2.add_internal_vertex_id(
        test_df, num2.renumbered_src_col_name, ["src", "src_old"]
    )


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_dask_pagerank(dask_client, directed):
    pandas.set_option("display.max_rows", 10000)

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
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
    g.from_cudf_edgelist(df, "src", "dst")

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    expected_pr = cugraph.pagerank(g)
    result_pr = dcg.pagerank(dg).compute()

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
    print("Mismatches:", err)
    assert err == 0


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("renumber", [False])
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_graph_renumber_false(renumber, dask_client, directed):
    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH / "karate.csv").as_posix()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    dg = cugraph.Graph(directed=directed)

    with pytest.raises(ValueError):
        dg.from_dask_cudf_edgelist(ddf, "src", "dst", renumber=renumber)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("renumber", [False])
@pytest.mark.parametrize("directed", IS_DIRECTED)
def test_multi_graph_renumber_false(renumber, dask_client, directed):
    input_data_path = (
        RAPIDS_DATASET_ROOT_DIR_PATH / "karate_multi_edge.csv"
    ).as_posix()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    dg = cugraph.MultiGraph(directed=directed)

    # ValueError always thrown since renumber must be True with
    # MNMG algorithms
    with pytest.raises(ValueError):
        dg.from_dask_cudf_edgelist(ddf, "src", "dst", renumber=renumber)


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize(
    "graph_file",
    utils.DATASETS_UNRENUMBERED,
    ids=[f"dataset={d.as_posix()}" for d in utils.DATASETS_UNRENUMBERED],
)
def test_mg_renumber_common_col_names(graph_file, dask_client):
    """
    Ensure that commonly-used column names in the input do not conflict with
    names used internally by NumberMap.
    """
    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    numbers = range(len(sources))
    offset_numbers = [n + 1 for n in numbers]
    floats = [float(n) for n in numbers]

    # test multi-column ("legacy" renumbering code path)
    gdf = cudf.DataFrame(
        {
            "src": numbers,
            "dst": numbers,
            "weights": floats,
            "col_a": sources,
            "col_b": sources,
            "col_c": destinations,
            "col_d": destinations,
        }
    )
    ddf = dask.dataframe.from_pandas(
        gdf, npartitions=len(dask_client.scheduler_info()["workers"])
    )

    renumbered_df, renumber_map = NumberMap.renumber(
        ddf, ["col_a", "col_b"], ["col_c", "col_d"]
    )

    assert renumber_map.renumbered_src_col_name != "src"
    assert renumber_map.renumbered_dst_col_name != "dst"
    assert renumber_map.renumbered_src_col_name in renumbered_df.columns
    assert renumber_map.renumbered_dst_col_name in renumbered_df.columns

    # test experimental renumbering code path
    gdf = cudf.DataFrame(
        {
            "src": numbers,
            "dst": offset_numbers,
            "weights": floats,
            "col_a": sources,
            "col_b": destinations,
        }
    )

    ddf = dask.dataframe.from_pandas(
        gdf, npartitions=len(dask_client.scheduler_info()["workers"])
    )

    renumbered_df, renumber_map = NumberMap.renumber(ddf, "col_a", "col_b")

    assert renumber_map.renumbered_src_col_name != "src"
    assert renumber_map.renumbered_dst_col_name != "dst"
    assert renumber_map.renumbered_src_col_name in renumbered_df.columns
    assert renumber_map.renumbered_dst_col_name in renumbered_df.columns


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
def test_pagerank_string_vertex_ids(dask_client):
    """
    Ensures string vertex IDs can be used.

    Note: the dask_client fixture sets up and tears down a LocalCUDACluster.
    See ../conftest.py
    """
    # Use pandas and to_csv() to create a CSV file that can be read in by both
    # dask_cudf and cudf.
    df = cudf.DataFrame(
        {
            "src": ["a1", "a1", "a2", "a3"],
            "dst": ["a2", "a3", "a4", "a4"],
        }
    )
    # SG
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(df, source="src", destination="dst")

    sg_results = cugraph.pagerank(G)
    sg_results = sg_results.sort_values("vertex").reset_index(drop=True)

    # MG
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    G_dask = cugraph.Graph(directed=True)
    G_dask.from_dask_cudf_edgelist(ddf, source="src", destination="dst")

    mg_results = dcg.pagerank(G_dask)
    # Organize results for easy comparison, this does not change the values. MG
    # Pagerank defaults to float64, so convert to float32 when comparing to SG
    mg_results = mg_results.compute().sort_values("vertex").reset_index(drop=True)
    mg_results["pagerank"] = mg_results["pagerank"].astype("float32")

    assert_frame_equal(sg_results, mg_results)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_mg_renumber_multi_column(dtype, dask_client):
    df = cudf.DataFrame(
        {"src_a": [i for i in range(0, 10)], "dst_a": [i for i in range(10, 20)]}
    ).astype(dtype)

    df["src_b"] = df["src_a"] + 10
    df["dst_b"] = df["dst_a"] + 20
    src_col = ["src_a", "src_b"]
    dst_col = ["dst_a", "dst_b"]

    ddf = dask_cudf.from_cudf(df, npartitions=2)
    edgelist_type = list(ddf.dtypes)
    G = cugraph.Graph()
    G.from_dask_cudf_edgelist(ddf, source=src_col, destination=dst_col)
    renumbered_ddf = G.edgelist.edgelist_df
    renumbered_edgelist_type = list(renumbered_ddf.dtypes)

    assert set(renumbered_edgelist_type).issubset(set(edgelist_type))
