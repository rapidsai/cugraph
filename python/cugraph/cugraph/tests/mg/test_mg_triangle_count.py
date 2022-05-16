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

import cudf
import cugraph
from cugraph.testing import utils
import cugraph.dask as dcg
import dask_cudf
import random
import warnings

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================


def setup_function():
    gc.collect()


# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx


HAS_START_LIST = [False, True]
IS_WEIGHTED = [False, True]


# =============================================================================
# Pytest fixtures
# =============================================================================

datasets = utils.DATASETS_UNDIRECTED

fixture_params = utils.genFixtureParamsProduct((datasets, "graph_file"),
                                               (HAS_START_LIST, "start_list"),
                                               (IS_WEIGHTED), "is_weighted"
                                               )


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    parameters = dict(zip(("graph_file",
                           "start_list",
                           "is_weighted"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the inputs and expected results from the triangle
    count algo.

    FIXME: We have to rely on nx for testing because the current cuGraph
    triangle count implementation only returns the totoal number of triangles
    instead of the number of triangles per vertices like nx.triangle or
    dcg.triangle_count
    """
    input_data_path = input_combo["graph_file"]
    is_weighted = input_combo["is_weighted"]

    M = utils.read_csv_for_nx(input_data_path)

    Gnx = nx.from_pandas_edgelist(
        M, source="0", target="1",
        edge_attr=is_weighted, create_using=nx.Graph()
    )

    start_list = input_combo["start_list"]
    if start_list:
        # sample k nodes from the nx graph
        k = random.randint(1, 10)
        start_list = random.sample(list(Gnx.nodes()), k)
    else:
        start_list = None

    nx_count_dic = nx.triangles(Gnx, start_list)
    nx_count = cudf.DataFrame()
    nx_count["vertex"] = nx_count_dic.keys()
    nx_count["counts"] = nx_count_dic.values()
    nx_count = nx_count.sort_values("vertex").reset_index(drop=True)

    # FIXME: convert start_list to a cudf.Series of dtype int32 because int64
    # might not be supported
    start_list = cudf.Series(start_list, dtype="int32")

    input_combo["nx_results"] = nx_count
    input_combo["start_list"] = start_list

    # Creating an edgelist from a dask cudf dataframe
    chunksize = dcg.get_chunksize(input_data_path)
    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )
    if is_weighted:
        edge_attr = "value"
    else:
        edge_attr = None

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(
        ddf, source='src', destination='dst',
        edge_attr=edge_attr, renumber=True)

    input_combo["MGGraph"] = dg

    return input_combo


@pytest.mark.parametrize("graph_file", utils.DATASETS_UNDIRECTED)
def test_triangles(dask_client, benchmark, input_expected_output):

    dg = input_expected_output["MGGraph"]
    start_list = input_expected_output["start_list"]

    result_counts = benchmark(dcg.triangle_count,
                              dg,
                              start_list)

    result_counts = result_counts.compute().sort_values(
        "vertex").reset_index(
            drop=True).rename(columns={"counts": "mg_counts"})

    expected_output = input_expected_output["nx_results"]

    # Update the dask cugraph triangle count with nx triangle count results
    # for easy comparison using cuDF DataFrame methods.
    result_counts["sg_counts"] = expected_output['counts']

    counts_diffs = result_counts.query('mg_counts != sg_counts')

    assert len(counts_diffs) == 0
