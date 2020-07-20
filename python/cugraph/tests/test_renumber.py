# Copyright (c) 2019, NVIDIA CORPORATION.
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

import numpy as np
import pandas as pd
import pytest

import cudf
from cugraph.structure.number_map import NumberMap
from cugraph.tests import utils

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask


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

    numbering = NumberMap()
    numbering.from_series(gdf["source_as_int"], gdf["dest_as_int"])
    src = numbering.to_internal_vertex_id(gdf["source_as_int"])
    dst = numbering.to_internal_vertex_id(gdf["dest_as_int"])

    check_src = numbering.from_internal_vertex_id(src)["0"]
    check_dst = numbering.from_internal_vertex_id(dst)["0"]

    assert check_src.equals(gdf["source_as_int"])
    assert check_dst.equals(gdf["dest_as_int"])


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

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ["source_as_int"], ["dest_as_int"])
    src = numbering.to_internal_vertex_id(gdf["source_as_int"])
    dst = numbering.to_internal_vertex_id(gdf["dest_as_int"])

    check_src = numbering.from_internal_vertex_id(src)["0"]
    check_dst = numbering.from_internal_vertex_id(dst)["0"]

    assert check_src.equals(gdf["source_as_int"])
    assert check_dst.equals(gdf["dest_as_int"])


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

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ["source_list"], ["dest_list"])
    src = numbering.to_internal_vertex_id(gdf["source_list"])
    dst = numbering.to_internal_vertex_id(gdf["dest_list"])

    check_src = numbering.from_internal_vertex_id(src)["0"]
    check_dst = numbering.from_internal_vertex_id(dst)["0"]

    assert check_src.equals(gdf["source_list"])
    assert check_dst.equals(gdf["dest_list"])


def test_renumber_negative():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.DataFrame.from_pandas(df[["source_list", "dest_list"]])

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ["source_list"], ["dest_list"])
    src = numbering.to_internal_vertex_id(gdf["source_list"])
    dst = numbering.to_internal_vertex_id(gdf["dest_list"])

    check_src = numbering.from_internal_vertex_id(src)["0"]
    check_dst = numbering.from_internal_vertex_id(dst)["0"]

    assert check_src.equals(gdf["source_list"])
    assert check_dst.equals(gdf["dest_list"])


def test_renumber_negative_col():
    source_list = [4, 6, 8, -20, 1]
    dest_list = [1, 29, 35, 0, 77]

    df = pd.DataFrame({"source_list": source_list, "dest_list": dest_list})

    gdf = cudf.DataFrame.from_pandas(df[["source_list", "dest_list"]])

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ["source_list"], ["dest_list"])
    src = numbering.to_internal_vertex_id(gdf["source_list"])
    dst = numbering.to_internal_vertex_id(gdf["dest_list"])

    check_src = numbering.from_internal_vertex_id(src)["0"]
    check_dst = numbering.from_internal_vertex_id(dst)["0"]

    assert check_src.equals(gdf["source_list"])
    assert check_dst.equals(gdf["dest_list"])


# Test all combinations of default/managed and pooled/non-pooled allocation


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

    numbering = NumberMap()
    numbering.from_series(df["src"], df["dst"])

    renumbered_df = numbering.add_internal_vertex_id(
        numbering.add_internal_vertex_id(df, "src_id", ["src"]),
        "dst_id", ["dst"]
    )

    check_src = numbering.from_internal_vertex_id(renumbered_df, "src_id")
    check_dst = numbering.from_internal_vertex_id(renumbered_df, "dst_id")

    assert check_src["src"].equals(check_src["0"])
    assert check_dst["dst"].equals(check_dst["0"])


# Test all combinations of default/managed and pooled/non-pooled allocation
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

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ["src"], ["dst"])

    renumbered_df = numbering.add_internal_vertex_id(
        numbering.add_internal_vertex_id(gdf, "src_id", ["src"]),
        "dst_id", ["dst"]
    )

    check_src = numbering.from_internal_vertex_id(renumbered_df, "src_id")
    check_dst = numbering.from_internal_vertex_id(renumbered_df, "dst_id")

    assert check_src["src"].equals(check_src["0"])
    assert check_dst["dst"].equals(check_dst["0"])


# Test all combinations of default/managed and pooled/non-pooled allocation
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

    numbering = NumberMap()
    numbering.from_dataframe(gdf, ["src", "src_old"], ["dst", "dst_old"])

    renumbered_df = numbering.add_internal_vertex_id(
        numbering.add_internal_vertex_id(
            gdf, "src_id", ["src", "src_old"]
        ),
        "dst_id",
        ["dst", "dst_old"],
    )

    check_src = numbering.from_internal_vertex_id(renumbered_df, "src_id")
    check_dst = numbering.from_internal_vertex_id(renumbered_df, "dst_id")

    assert check_src["src"].equals(check_src["0"])
    assert check_src["src_old"].equals(check_src["1"])
    assert check_dst["dst"].equals(check_dst["0"])
    assert check_dst["dst_old"].equals(check_dst["1"])


@pytest.fixture()
def client_setup():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    client


# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_opg_renumber(graph_file, client_setup):
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
@pytest.mark.parametrize("graph_file", utils.DATASETS)
def test_opg_renumber2(graph_file, client_setup):
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
    check_src = check_src.sort_values('weight').reset_index(drop=True)
    check_dst = num2.from_internal_vertex_id(ren2, "dst").compute()
    check_dst = check_dst.sort_values('weight').reset_index(drop=True)

    assert check_src["0"].to_pandas().equals(gdf["src"].to_pandas())
    assert check_src["1"].to_pandas().equals(gdf["src_old"].to_pandas())
    assert check_dst["0"].to_pandas().equals(gdf["dst"].to_pandas())
    assert check_dst["1"].to_pandas().equals(gdf["dst_old"].to_pandas())
