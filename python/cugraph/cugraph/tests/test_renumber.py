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

import pandas as pd
import pytest
import cudf
from cudf.testing import assert_series_equal

from cugraph.structure.number_map import NumberMap
from cugraph.tests import utils


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

    renumbered_gdf, renumber_map = NumberMap.renumber(gdf,
                                                      "source_as_int",
                                                      "dest_as_int")

    check_src = renumber_map.from_internal_vertex_id(renumbered_gdf['src']
                                                     )["0"]
    check_dst = renumber_map.from_internal_vertex_id(renumbered_gdf['dst']
                                                     )["0"]

    assert_series_equal(check_src, gdf["source_as_int"], check_names=False)
    assert_series_equal(check_dst, gdf["dest_as_int"], check_names=False)


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

    renumbered_gdf, renumber_map = NumberMap.renumber(gdf,
                                                      ["source_as_int"],
                                                      ["dest_as_int"])

    check_src = renumber_map.from_internal_vertex_id(renumbered_gdf['src']
                                                     )["0"]
    check_dst = renumber_map.from_internal_vertex_id(renumbered_gdf['dst']
                                                     )["0"]

    assert_series_equal(check_src, gdf["source_as_int"], check_names=False)
    assert_series_equal(check_dst, gdf["dest_as_int"], check_names=False)


@pytest.mark.skip(reason="temporarily dropped string support")
def test_renumber_ips_str_cols():

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

    renumbered_gdf, renumber_map = NumberMap.renumber(gdf,
                                                      ["source_as_int"],
                                                      ["dest_as_int"])

    check_src = renumber_map.from_internal_vertex_id(renumbered_gdf['src']
                                                     )["0"]
    check_dst = renumber_map.from_internal_vertex_id(renumbered_gdf['dst']
                                                     )["0"]

    assert_series_equal(check_src, gdf["source_list"], check_names=False)
    assert_series_equal(check_dst, gdf["dest_list"], check_names=False)


def test_renumber_negative():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.DataFrame.from_pandas(df[["source_list", "dest_list"]])

    renumbered_gdf, renumber_map = NumberMap.renumber(gdf,
                                                      "source_list",
                                                      "dest_list")

    check_src = renumber_map.from_internal_vertex_id(renumbered_gdf['src']
                                                     )["0"]
    check_dst = renumber_map.from_internal_vertex_id(renumbered_gdf['dst']
                                                     )["0"]

    assert_series_equal(check_src, gdf["source_list"], check_names=False)
    assert_series_equal(check_dst, gdf["dest_list"], check_names=False)


def test_renumber_negative_col():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.DataFrame.from_pandas(df[["source_list", "dest_list"]])

    renumbered_gdf, renumber_map = NumberMap.renumber(gdf,
                                                      "source_list",
                                                      "dest_list")

    check_src = renumber_map.from_internal_vertex_id(renumbered_gdf['src']
                                                     )["0"]
    check_dst = renumber_map.from_internal_vertex_id(renumbered_gdf['dst']
                                                     )["0"]

    assert_series_equal(check_src, gdf["source_list"], check_names=False)
    assert_series_equal(check_dst, gdf["dest_list"], check_names=False)


@pytest.mark.skip(reason="dropped renumbering from series support")
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_renumber_series(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    df = cudf.DataFrame()
    df["src"] = cudf.Series([x + translate for x in sources.
                            values_host])
    df["dst"] = cudf.Series([x + translate for x in destinations.
                            values_host])

    numbering_series_1 = NumberMap()
    numbering_series_1.from_series(df["src"])

    numbering_series_2 = NumberMap()
    numbering_series_2.from_series(df["dst"])

    renumbered_src = numbering_series_1.add_internal_vertex_id(
        df["src"], "src_id")
    renumbered_dst = numbering_series_2.add_internal_vertex_id(
        df["dst"], "dst_id")

    check_src = numbering_series_1.from_internal_vertex_id(renumbered_src,
                                                           "src_id")
    check_dst = numbering_series_2.from_internal_vertex_id(renumbered_dst,
                                                           "dst_id")

    assert_series_equal(check_src["0_y"], check_src["0_x"], check_names=False)
    assert_series_equal(check_dst["0_y"], check_dst["0_x"], check_names=False)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_renumber_files(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    df = cudf.DataFrame()
    df["src"] = cudf.Series([x + translate for x in sources.
                            values_host])
    df["dst"] = cudf.Series([x + translate for x in destinations.
                            values_host])

    exp_src = cudf.Series([x + translate for x in sources.
                          values_host])
    exp_dst = cudf.Series([x + translate for x in destinations.
                          values_host])

    renumbered_df, renumber_map = NumberMap.renumber(df, "src", "dst",
                                                     preserve_order=True)

    unrenumbered_df = renumber_map.unrenumber(renumbered_df, "src",
                                              preserve_order=True)
    unrenumbered_df = renumber_map.unrenumber(unrenumbered_df, "dst",
                                              preserve_order=True)

    assert_series_equal(exp_src, unrenumbered_df["src"], check_names=False)
    assert_series_equal(exp_dst, unrenumbered_df["dst"], check_names=False)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_renumber_files_col(graph_file):
    gc.collect()

    M = utils.read_csv_for_nx(graph_file)
    sources = cudf.Series(M["0"])
    destinations = cudf.Series(M["1"])

    translate = 1000

    gdf = cudf.DataFrame()
    gdf['src'] = cudf.Series([x + translate for x in sources.values_host])
    gdf['dst'] = cudf.Series([x + translate for x in destinations.
                             values_host])

    exp_src = cudf.Series([x + translate for x in sources.
                          values_host])
    exp_dst = cudf.Series([x + translate for x in destinations.
                          values_host])

    renumbered_df, renumber_map = NumberMap.renumber(gdf, ["src"], ["dst"],
                                                     preserve_order=True)

    unrenumbered_df = renumber_map.unrenumber(renumbered_df, "src",
                                              preserve_order=True)
    unrenumbered_df = renumber_map.unrenumber(unrenumbered_df, "dst",
                                              preserve_order=True)

    assert_series_equal(exp_src, unrenumbered_df["src"], check_names=False)
    assert_series_equal(exp_dst, unrenumbered_df["dst"], check_names=False)


@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_renumber_files_multi_col(graph_file):
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

    renumbered_df, renumber_map = NumberMap.renumber(gdf,
                                                     ["src", "src_old"],
                                                     ["dst", "dst_old"],
                                                     preserve_order=True)

    unrenumbered_df = renumber_map.unrenumber(renumbered_df, "src",
                                              preserve_order=True)
    unrenumbered_df = renumber_map.unrenumber(unrenumbered_df, "dst",
                                              preserve_order=True)

    assert_series_equal(gdf["src"], unrenumbered_df["0_src"],
                        check_names=False)
    assert_series_equal(gdf["src_old"], unrenumbered_df["1_src"],
                        check_names=False)
    assert_series_equal(gdf["dst"], unrenumbered_df["0_dst"],
                        check_names=False)
    assert_series_equal(gdf["dst_old"], unrenumbered_df["1_dst"],
                        check_names=False)
