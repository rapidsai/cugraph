# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import pytest

import cugraph.dask as dcg

import cugraph
import dask_cudf
from cugraph.testing import utils

# from cugraph.dask.common.mg_utils import is_single_gpu

try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print(
        "\n\nWARNING: rapids_pytest_benchmark is not installed, "
        "falling back to pytest_benchmark fixtures.\n"
    )

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass


# =============================================================================
# Parameters
# =============================================================================
DATASETS_ASYMMETRIC = [utils.RAPIDS_DATASET_ROOT_DIR_PATH / "karate-asymmetric.csv"]


###############################################################################
# Fixtures
# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
@pytest.fixture(
    scope="module",
    params=DATASETS_ASYMMETRIC,
    ids=[f"dataset={d.as_posix()}" for d in DATASETS_ASYMMETRIC],
)
def daskGraphFromDataset(request, dask_client):
    """
    Returns a new dask dataframe created from the dataset file param.
    This creates a directed Graph.
    """
    # Since parameterized fixtures do not assign param names to param values,
    # manually call the helper to do so.
    setFixtureParamNames(request, ["dataset"])
    dataset = request.param

    chunksize = dcg.get_chunksize(dataset)
    ddf = dask_cudf.read_csv(
        dataset,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=True)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "value")
    return dg


@pytest.fixture(
    scope="module",
    params=utils.DATASETS_UNDIRECTED,
    ids=[f"dataset={d.as_posix()}" for d in utils.DATASETS_UNDIRECTED],
)
def uddaskGraphFromDataset(request, dask_client):
    """
    Returns a new dask dataframe created from the dataset file param.
    This creates an undirected Graph.
    """
    # Since parameterized fixtures do not assign param names to param
    # values, manually call the helper to do so.
    setFixtureParamNames(request, ["dataset"])
    dataset = request.param

    chunksize = dcg.get_chunksize(dataset)
    ddf = dask_cudf.read_csv(
        dataset,
        chunksize=chunksize,
        delimiter=" ",
        names=["src", "dst", "value"],
        dtype=["int32", "int32", "float32"],
    )

    dg = cugraph.Graph(directed=False)
    dg.from_dask_cudf_edgelist(ddf, "src", "dst", "value")
    return dg


###############################################################################
# Tests
# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
# FIXME: Implement more robust tests
def test_mg_louvain_with_edgevals_directed_graph(daskGraphFromDataset):
    # Directed graphs are not supported by Louvain and a ValueError should be
    # raised
    with pytest.raises(ValueError):
        parts, mod = dcg.louvain(daskGraphFromDataset)


###############################################################################
# Tests
# @pytest.mark.skipif(
#    is_single_gpu(), reason="skipping MG testing on Single GPU system"
# )
# FIXME: Implement more robust tests
def test_mg_louvain_with_edgevals_undirected_graph(uddaskGraphFromDataset):
    parts, mod = dcg.louvain(uddaskGraphFromDataset)

    # FIXME: either call Nx with the same dataset and compare results, or
    # hardcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()
