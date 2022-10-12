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

import pandas as pd
import pytest
import cudf
from cudf.testing import assert_series_equal

from cugraph.structure.number_map import NumberMap
from cugraph.testing import utils
from cugraph.experimental.datasets import DATASETS


def test_renumber_ips():
    source_list = [
        "192.168.1.1",
        "172.217.5.238",
        "216.228.121.209",
        "192.16.31.23",
    ]
    dest_list = [
        "172.217.5.238",
        "216.228.121.209",
        "192.16.31.23",
        "192.168.1.1",
    ]

    pdf = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.from_pandas(pdf)

    gdf["source_as_int"] = gdf["source_list"].str.ip2int()
    gdf["dest_as_int"] = gdf["dest_list"].str.ip2int()

    renumbered_gdf, renumber_map = NumberMap.renumber(
        gdf, "source_as_int", "dest_as_int", preserve_order=True
    )

    input_check = renumbered_gdf.merge(gdf, on=["source_list", "dest_list"])

    output_check = renumber_map.from_internal_vertex_id(
        renumbered_gdf,
        renumber_map.renumbered_src_col_name,
        external_column_names=["check_src"],
    )
    output_check = renumber_map.from_internal_vertex_id(
        output_check,
        renumber_map.renumbered_dst_col_name,
        external_column_names=["check_dst"],
    )

    merged = output_check.merge(input_check, on=["source_list", "dest_list"])

    assert_series_equal(merged["check_src"], merged["source_as_int"], check_names=False)
    assert_series_equal(merged["check_dst"], merged["dest_as_int"], check_names=False)


def test_renumber_ips_cols():

    source_list = [
        "192.168.1.1",
        "172.217.5.238",
        "216.228.121.209",
        "192.16.31.23",
    ]
    dest_list = [
        "172.217.5.238",
        "216.228.121.209",
        "192.16.31.23",
        "192.168.1.1",
    ]

    pdf = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.from_pandas(pdf)

    gdf["source_as_int"] = gdf["source_list"].str.ip2int()
    gdf["dest_as_int"] = gdf["dest_list"].str.ip2int()

    renumbered_gdf, renumber_map = NumberMap.renumber(
        gdf, ["source_as_int"], ["dest_as_int"], preserve_order=True
    )

    input_check = renumbered_gdf.merge(gdf, on=["source_list", "dest_list"])

    output_check = renumber_map.from_internal_vertex_id(
        renumbered_gdf,
        renumber_map.renumbered_src_col_name,
        external_column_names=["check_src"],
    )
    output_check = renumber_map.from_internal_vertex_id(
        output_check,
        renumber_map.renumbered_dst_col_name,
        external_column_names=["check_dst"],
    )

    merged = output_check.merge(input_check, on=["source_list", "dest_list"])

    assert_series_equal(merged["check_src"], merged["source_as_int"], check_names=False)
    assert_series_equal(merged["check_dst"], merged["dest_as_int"], check_names=False)


def test_renumber_negative():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.DataFrame.from_pandas(df[["source_list", "dest_list"]])
    gdf["original_src"] = gdf["source_list"]
    gdf["original_dst"] = gdf["dest_list"]

    renumbered_gdf, renumber_map = NumberMap.renumber(
        gdf, "source_list", "dest_list", preserve_order=True
    )

    input_check = renumbered_gdf.merge(gdf, on=["original_src", "original_dst"])

    output_check = renumber_map.from_internal_vertex_id(
        renumbered_gdf,
        renumber_map.renumbered_src_col_name,
        external_column_names=["check_src"],
    )
    output_check = renumber_map.from_internal_vertex_id(
        output_check,
        renumber_map.renumbered_dst_col_name,
        external_column_names=["check_dst"],
    )

    merged = output_check.merge(input_check, on=["original_src", "original_dst"])

    assert_series_equal(merged["check_src"], merged["original_src"], check_names=False)
    assert_series_equal(merged["check_dst"], merged["original_dst"], check_names=False)


def test_renumber_negative_col():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.DataFrame.from_pandas(df[["source_list", "dest_list"]])
    gdf["original_src"] = gdf["source_list"]
    gdf["original_dst"] = gdf["dest_list"]

    renumbered_gdf, renumber_map = NumberMap.renumber(
        gdf, ["source_list"], ["dest_list"], preserve_order=True
    )

    input_check = renumbered_gdf.merge(gdf, on=["original_src", "original_dst"])

    output_check = renumber_map.from_internal_vertex_id(
        renumbered_gdf,
        renumber_map.renumbered_src_col_name,
        external_column_names=["check_src"],
    )
    output_check = renumber_map.from_internal_vertex_id(
        output_check,
        renumber_map.renumbered_dst_col_name,
        external_column_names=["check_dst"],
    )

    merged = output_check.merge(input_check, on=["original_src", "original_dst"])

    assert_series_equal(merged["check_src"], merged["original_src"], check_names=False)
    assert_series_equal(merged["check_dst"], merged["original_dst"], check_names=False)


@pytest.mark.parametrize("graph_file", DATASETS)
def test_renumber_files(graph_file):
    gc.collect()
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    df = cudf.DataFrame()
    df["src"] = cudf.Series([x + translate for x in sources.values_host])
    df["dst"] = cudf.Series([x + translate for x in destinations.values_host])

    exp_src = cudf.Series([x + translate for x in sources.values_host])
    exp_dst = cudf.Series([x + translate for x in destinations.values_host])

    renumbered_df, renumber_map = NumberMap.renumber(
        df, "src", "dst", preserve_order=True
    )

    unrenumbered_df = renumber_map.unrenumber(
        renumbered_df, renumber_map.renumbered_src_col_name, preserve_order=True
    )
    unrenumbered_df = renumber_map.unrenumber(
        unrenumbered_df, renumber_map.renumbered_dst_col_name, preserve_order=True
    )

    assert_series_equal(
        exp_src,
        unrenumbered_df[renumber_map.renumbered_src_col_name],
        check_names=False,
    )
    assert_series_equal(
        exp_dst,
        unrenumbered_df[renumber_map.renumbered_dst_col_name],
        check_names=False,
    )


@pytest.mark.parametrize("graph_file", DATASETS)
def test_renumber_files_col(graph_file):
    gc.collect()
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf["src"] = cudf.Series([x + translate for x in sources.values_host])
    gdf["dst"] = cudf.Series([x + translate for x in destinations.values_host])

    exp_src = cudf.Series([x + translate for x in sources.values_host])
    exp_dst = cudf.Series([x + translate for x in destinations.values_host])

    renumbered_df, renumber_map = NumberMap.renumber(
        gdf, ["src"], ["dst"], preserve_order=True
    )

    unrenumbered_df = renumber_map.unrenumber(
        renumbered_df, renumber_map.renumbered_src_col_name, preserve_order=True
    )
    unrenumbered_df = renumber_map.unrenumber(
        unrenumbered_df, renumber_map.renumbered_dst_col_name, preserve_order=True
    )

    assert_series_equal(
        exp_src,
        unrenumbered_df[renumber_map.renumbered_src_col_name],
        check_names=False,
    )
    assert_series_equal(
        exp_dst,
        unrenumbered_df[renumber_map.renumbered_dst_col_name],
        check_names=False,
    )


@pytest.mark.parametrize("graph_file", DATASETS)
def test_renumber_files_multi_col(graph_file):
    gc.collect()
    dataset_path = graph_file.get_path()
    M = utils.read_csv_for_nx(dataset_path)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf["src_old"] = sources
    gdf["dst_old"] = destinations
    gdf["src"] = sources + translate
    gdf["dst"] = destinations + translate

    renumbered_df, renumber_map = NumberMap.renumber(
        gdf, ["src", "src_old"], ["dst", "dst_old"], preserve_order=True
    )

    unrenumbered_df = renumber_map.unrenumber(
        renumbered_df, renumber_map.renumbered_src_col_name, preserve_order=True
    )
    unrenumbered_df = renumber_map.unrenumber(
        unrenumbered_df, renumber_map.renumbered_dst_col_name, preserve_order=True
    )

    src = renumber_map.renumbered_src_col_name
    dst = renumber_map.renumbered_dst_col_name
    assert_series_equal(gdf["src"], unrenumbered_df[f"0_{src}"], check_names=False)
    assert_series_equal(gdf["src_old"], unrenumbered_df[f"1_{src}"], check_names=False)
    assert_series_equal(gdf["dst"], unrenumbered_df[f"0_{dst}"], check_names=False)
    assert_series_equal(gdf["dst_old"], unrenumbered_df[f"1_{dst}"], check_names=False)


def test_renumber_common_col_names():
    """
    Ensure that commonly-used column names in the input do not conflict with
    names used internally by NumberMap.
    """
    # test multi-column ("legacy" renumbering code path)
    gdf = cudf.DataFrame(
        {
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "weights": [0.1, 0.2, 0.3],
            "col_a": [8, 1, 82],
            "col_b": [1, 82, 3],
            "col_c": [9, 7, 2],
            "col_d": [1, 2, 3],
        }
    )

    renumbered_df, renumber_map = NumberMap.renumber(
        gdf, ["col_a", "col_b"], ["col_c", "col_d"]
    )

    assert renumber_map.renumbered_src_col_name != "src"
    assert renumber_map.renumbered_dst_col_name != "dst"
    assert renumber_map.renumbered_src_col_name in renumbered_df.columns
    assert renumber_map.renumbered_dst_col_name in renumbered_df.columns

    # test experimental renumbering code path
    gdf = cudf.DataFrame(
        {
            "src": [0, 1, 2],
            "dst": [1, 2, 3],
            "weights": [0.1, 0.2, 0.3],
            "col_a": [0, 1, 2],
            "col_b": [1, 2, 3],
        }
    )

    renumbered_df, renumber_map = NumberMap.renumber(gdf, "col_a", "col_b")

    assert renumber_map.renumbered_src_col_name != "src"
    assert renumber_map.renumbered_dst_col_name != "dst"
    assert renumber_map.renumbered_src_col_name in renumbered_df.columns
    assert renumber_map.renumbered_dst_col_name in renumbered_df.columns


def test_renumber_unrenumber_non_default_vert_names():
    """
    Test that renumbering a dataframe with generated src/dst column names can
    be used for unrenumbering results.
    """
    input_gdf = cudf.DataFrame(
        {
            "dst": [1, 2, 3],
            "weights": [0.1, 0.2, 0.3],
            "col_a": [99, 199, 2],
            "col_b": [199, 2, 32],
        }
    )

    renumbered_df, number_map = NumberMap.renumber(input_gdf, "col_a", "col_b")

    some_result_gdf = cudf.DataFrame({"vertex": [0, 1, 2, 3]})
    expected_values = [99, 199, 2, 32]

    some_result_gdf = number_map.unrenumber(some_result_gdf, "vertex")

    assert sorted(expected_values) == sorted(
        some_result_gdf["vertex"].to_arrow().to_pylist()
    )
