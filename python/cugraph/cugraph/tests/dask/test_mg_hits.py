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

import cugraph.dask as dcg
import gc
import pytest
import cugraph
import dask_cudf
import cudf
from cugraph.dask.common.mg_utils import is_single_gpu
from cugraph.tests.utils import RAPIDS_DATASET_ROOT_DIR_PATH
from cugraph.tests import utils

# =============================================================================
# Pytest Setup / Teardown - called for each test function
# =============================================================================
def setup_function():
    gc.collect()


# =============================================================================
# Pytest fixtures
# =============================================================================
datasets = utils.DATASETS_UNDIRECTED + \
           [utils.RAPIDS_DATASET_ROOT_DIR_PATH/"email-Eu-core.csv"]
fixture_params = utils.genFixtureParamsProduct((datasets, "graph_file"),
                                               ([50], "max_iter"),
                                               ([1.0e-6], "tol"),
                                               )


@pytest.fixture(scope="module", params=fixture_params)
def input_combo(request):
    """
    Simply return the current combination of params as a dictionary for use in
    tests or other parameterized fixtures.
    """
    print("parameters are \n", request.param, flush=True)
    parameters = dict(zip(("graph_file", "max_iter", "tol"), request.param))

    return parameters


@pytest.fixture(scope="module")
def input_expected_output(input_combo):
    """
    This fixture returns the expected results from the HITS algo.(based on cuGraph
    HITS) which can be used for validation.
    """

    G = utils.generate_cugraph_graph_from_file(
        input_combo["graph_file"])
    sg_cugraph_hits = cugraph.hits(
                            G,
                            input_combo["max_iter"],
                            input_combo["tol"])
    sg_cugraph_hits = sg_cugraph_hits.sort_values("vertex").reset_index(drop=True)

    input_combo["sg_cugraph_results"] = sg_cugraph_hits
    return input_combo


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)

def test_cugraph_hits(benchmark, input_combo):
    """
    Simply run cuGraph HITS on the same set of input combinations used for the
    cuGraph HITS tests.
    This is only in place for generating comparison performance numbers.
    """

    G = utils.generate_cugraph_graph_from_file(
        input_combo["graph_file"])
    
    sg_cugraph_hits = benchmark(cugraph.hits,
                             G,
                             input_combo["max_iter"],
                             input_combo["tol"])


@pytest.mark.skipif(
    is_single_gpu(), reason="skipping MG testing on Single GPU system"
)
def test_dask_hits(dask_client, benchmark, input_expected_output):
        
    input_data_path = input_expected_output["graph_file"]
    print(f"dataset={input_data_path}")
    chunksize = dcg.get_chunksize(input_expected_output["graph_file"])

    ddf = dask_cudf.read_csv(
        input_data_path,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst")
    


    result_dist = benchmark(dcg.hits,
                            dg,
                            input_expected_output["max_iter"],
                            input_expected_output["tol"])
    result_dist = result_dist.compute().sort_values("vertex").reset_index(drop=True)

    expected_output = input_expected_output["sg_cugraph_results"].sort_values("vertex").reset_index(drop=True)

    # Update the dask cugraph HITS results with sg cugraph results for easy comparison using
    # cuDF DataFrame methods.
    results_dist["cugraph_hubs"] = expected_output['hubs']
    results_dist["cugraph_authorities"] = expected_output["authorities"]

    hubs_diffs1 = cugraph_hits.query('hubs - nx_hubs > 0.00001')
    hubs_diffs2 = cugraph_hits.query('hubs - nx_hubs < -0.00001')
    authorities_diffs1 = cugraph_hits.query(
        'authorities - nx_authorities > 0.0001')
    authorities_diffs2 = cugraph_hits.query(
        'authorities - nx_authorities < -0.0001')

    assert len(hubs_diffs1) == 0
    assert len(hubs_diffs2) == 0
    assert len(authorities_diffs1) == 0
    assert len(authorities_diffs2) == 0


