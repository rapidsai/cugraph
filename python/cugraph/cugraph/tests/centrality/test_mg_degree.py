# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
from cugraph.testing.utils import RAPIDS_DATASET_ROOT_DIR_PATH

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


IS_DIRECTED = [True, False]

DATA_PATH = [
    (RAPIDS_DATASET_ROOT_DIR_PATH / "karate-asymmetric.csv").as_posix(),
    (RAPIDS_DATASET_ROOT_DIR_PATH / "polbooks.csv").as_posix(),
    (RAPIDS_DATASET_ROOT_DIR_PATH / "email-Eu-core.csv").as_posix(),
]


@pytest.mark.skipif(is_single_gpu(), reason="skipping MG testing on Single GPU system")
@pytest.mark.parametrize("directed", IS_DIRECTED)
@pytest.mark.parametrize("data_file", DATA_PATH)
def test_dask_mg_degree(dask_client, directed, data_file):

    input_data_path = data_file
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

    dg = cugraph.Graph(directed=directed)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")
    dg.compute_renumber_edge_list()

    g = cugraph.Graph(directed=directed)
    g.from_cudf_edgelist(df, "src", "dst")

    merge_df_in_degree = (
        dg.in_degree()
        .merge(g.in_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    merge_df_out_degree = (
        dg.out_degree()
        .merge(g.out_degree(), on="vertex", suffixes=["_dg", "_g"])
        .compute()
    )

    merge_df_degree = (
        dg.degree().merge(g.degree(), on="vertex", suffixes=["_dg", "_g"]).compute()
    )

    assert_series_equal(
        merge_df_in_degree["degree_dg"],
        merge_df_in_degree["degree_g"],
        check_names=False,
        check_dtype=False,
    )

    assert_series_equal(
        merge_df_out_degree["degree_dg"],
        merge_df_out_degree["degree_g"],
        check_names=False,
        check_dtype=False,
    )

    assert_series_equal(
        merge_df_degree["degree_dg"],
        merge_df_degree["degree_g"],
        check_names=False,
        check_dtype=False,
    )
