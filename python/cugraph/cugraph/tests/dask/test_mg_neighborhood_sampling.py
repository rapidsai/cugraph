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
def test_mg_neighborhood_sampling_simple(dask_client):

    from cugraph.experimental.dask import uniform_neighborhood_sampling

    df = cudf.DataFrame({"src": cudf.Series([0, 1, 1, 2, 2, 2, 3, 4],
                                            dtype="int32"),
                         "dst": cudf.Series([1, 3, 4, 0, 1, 3, 5, 5],
                                            dtype="int32"),
                         "value": cudf.Series([0.1, 2.1, 1.1, 5.1, 3.1,
                                               4.1, 7.2, 3.2],
                                              dtype="float32"),
                         })
    ddf = dask_cudf.from_cudf(df, npartitions=2)

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(ddf, "src", "dst", "value")

    # TODO: Incomplete, include more testing for tree graph as well as
    # for larger graphs
    start_list = cudf.Series([0, 1], dtype="int32")
    info_list = cudf.Series([0, 0], dtype="int32")
    fanout_vals = [1, 1]
    with_replacement = True
    result_nbr = uniform_neighborhood_sampling(G,
                                               (start_list, info_list),
                                               fanout_vals,
                                               with_replacement)
    result_nbr = result_nbr.compute()

    # Since the validity of results have (probably) been tested at botht he C++
    # and C layers, simply test that the python interface and conversions were
    # done correctly.
    assert result_nbr['sources'].dtype == "int32"
    assert result_nbr['destinations'].dtype == "int32"
    assert result_nbr['labels'].dtype == "int32"
    assert result_nbr['indices'].dtype == "int32"

    # ALl labels should be 0 or 1
    assert result_nbr['labels'].isin([0, 1]).all()


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_mg_neighborhood_sampling_tree(dask_client):

    from cugraph.experimental.dask import uniform_neighborhood_sampling

    input_data_path = (utils.RAPIDS_DATASET_ROOT_DIR_PATH /
                       "small_tree.csv").as_posix()
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    G = cugraph.Graph(directed=True)
    G.from_dask_cudf_edgelist(ddf, "src", "dst", "value")

    # TODO: Incomplete, include more testing for tree graph as well as
    # for larger graphs
    start_list = cudf.Series([0, 0], dtype="int32")
    info_list = cudf.Series([0, 0], dtype="int32")
    fanout_vals = [4, 1, 3]
    with_replacement = True
    result_nbr = uniform_neighborhood_sampling(G,
                                               (start_list, info_list),
                                               fanout_vals,
                                               with_replacement)
    result_nbr = result_nbr.compute()

    # Since the validity of results have (probably) been tested at botht he C++
    # and C layers, simply test that the python interface and conversions were
    # done correctly.
    assert result_nbr['sources'].dtype == "int32"
    assert result_nbr['destinations'].dtype == "int32"
    assert result_nbr['labels'].dtype == "int32"
    assert result_nbr['indices'].dtype == "int32"

    # All labels should be 0
    assert (result_nbr['labels'] == 0).all()
