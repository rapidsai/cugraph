# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import cugraph.dask as dcg
import gc
import pytest
import cugraph
import dask_cudf
import cudf
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.tests.utils import RAPIDS_DATASET_ROOT_DIR_PATH


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_bfs(dask_client):
    gc.collect()

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "netscience.csv").as_posix()

    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    def modify_dataset(df):
        temp_df = cudf.DataFrame()
        temp_df['src'] = df['src']+1000
        temp_df['dst'] = df['dst']+1000
        temp_df['value'] = df['value']
        return cudf.concat([df, temp_df])

    meta = ddf._meta
    ddf = ddf.map_partitions(modify_dataset, meta=meta)

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    df = modify_dataset(df)

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, "src", "dst")

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    expected_dist = cugraph.bfs(g, [0, 1000])
    result_dist = dcg.bfs(dg, [0, 1000])
    result_dist = result_dist.compute()

    compare_dist = expected_dist.merge(
        result_dist, on="vertex", suffixes=["_local", "_dask"]
    )

    err = 0

    for i in range(len(compare_dist)):
        if (
            compare_dist["distance_local"].iloc[i]
            != compare_dist["distance_dask"].iloc[i]
        ):
            err = err + 1
    assert err == 0


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_bfs_multi_column_depthlimit(dask_client):
    gc.collect()

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "netscience.csv").as_posix()
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src_a", "dst_a", "value"],
        dtype=["int32", "int32", "float32"],
    )
    ddf['src_b'] = ddf['src_a'] + 1000
    ddf['dst_b'] = ddf['dst_a'] + 1000

    df = cudf.read_csv(
        input_data_path,
        delimiter=" ",
        names=["src_a", "dst_a", "value"],
        dtype=["int32", "int32", "float32"],
    )
    df['src_b'] = df['src_a'] + 1000
    df['dst_b'] = df['dst_a'] + 1000

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, ["src_a", "src_b"], ["dst_a", "dst_b"])

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, ["src_a", "src_b"], ["dst_a", "dst_b"])

    start = cudf.DataFrame()
    start['a'] = [0]
    start['b'] = [1000]

    depth_limit = 18
    expected_dist = cugraph.bfs(g, start, depth_limit=depth_limit)
    result_dist = dcg.bfs(dg, start, depth_limit=depth_limit)
    result_dist = result_dist.compute()

    compare_dist = expected_dist.merge(
        result_dist, on=["0_vertex", "1_vertex"], suffixes=["_local", "_dask"]
    )

    err = 0
    for i in range(len(compare_dist)):
        if (
            compare_dist["distance_local"].iloc[i] <= depth_limit and
            compare_dist["distance_dask"].iloc[i] <= depth_limit and
            compare_dist["distance_local"].iloc[i]
            != compare_dist["distance_dask"].iloc[i]
        ):
            err = err + 1
    assert err == 0
