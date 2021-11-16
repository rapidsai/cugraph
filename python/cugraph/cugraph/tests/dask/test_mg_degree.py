# Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

import gc
import pytest
import cudf
import dask_cudf
from cudf.testing import assert_series_equal

import cugraph
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.tests.utils import RAPIDS_DATASET_ROOT_DIR_PATH


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_mg_degree(dask_client):
    gc.collect()

    input_data_path = (RAPIDS_DATASET_ROOT_DIR_PATH /
                       "karate-asymmetric.csv").as_posix()
    print(f"dataset={input_data_path}")

    chunksize = cugraph.dask.get_chunksize(input_data_path)

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

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, "src", "dst")

    merge_df_in = (
        dg.in_degree()
        .merge(g.in_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    merge_df_out = (
        dg.out_degree()
        .merge(g.out_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    assert_series_equal(merge_df_in["degree_dg"], merge_df_in["degree_g"],
                        check_names=False)
    assert_series_equal(merge_df_out["degree_dg"], merge_df_out["degree_g"],
                        check_names=False)
