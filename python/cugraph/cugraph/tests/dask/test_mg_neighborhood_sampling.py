# Copyright (c) 2022, NVIDIA CORPORATION.
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
import cugraph.dask as dcg
import cugraph
import dask_cudf
import cudf
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.tests import utils


# =============================================================================
# Test helpers
# =============================================================================
def setup_function():
    gc.collect()


# datasets = utils.RAPIDS_DATASET_ROOT_DIR_PATH/"karate.csv"
datasets = utils.DATASETS_SMALL
fixture_params = utils.genFixtureParamsProduct((datasets, "graph_file"))


def _get_param_args(param_name, param_values):
    """
    Returns a tuple of (<param_name>, <pytest.param list>) which can be applied
    as the args to pytest.mark.parametrize(). The pytest.param list also
    contains param id string formed from the param name and values.
    """
    return (param_name,
            [pytest.param(v, id=f"{param_name}={v}") for v in param_values])


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_mg_neighborhood_sampling_tree(dask_client):
    gc.collect()

    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH /
                       "small_tree.csv").as_posix()
    print(f"dataset={input_data_path}")
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
    g.from_cudf_edgelist(df, "src", "dst", "value")

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "value")

    # TODO: Incomplete
    start_list = cudf.Series([0, 0])
    info_list = cudf.Series([0, 0])
    fanout_vals = cudf.Series([4, 1, 3])
    with_replacement = True
    result_nbr = dcg.uniform_neighborhood(dg, start_list,
                                          info_list,
                                          fanout_vals,
                                          with_replacement)
    result_nbr = result_nbr.compute()

    # Test that lengths of outputs are as intended for small graph
    assert len(result_nbr['srcs']) == 10
    assert len(result_nbr['dsts']) == 10
    assert len(result_nbr['labels']) == 2
    assert len(result_nbr['index']) == 2

    # Because there is no SG version of neighborhood sampling, unsure
    # as to what to compare, for now the test should fail always
    assert False
