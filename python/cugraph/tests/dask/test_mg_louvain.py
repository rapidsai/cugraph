# Copyright (c) 2020, NVIDIA CORPORATION.
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
import cugraph.comms as Comms
from dask.distributed import Client
import cugraph
import dask_cudf
from dask_cuda import LocalCUDACluster
from cugraph.tests import utils

try:
    from rapids_pytest_benchmark import setFixtureParamNames
except ImportError:
    print("\n\nWARNING: rapids_pytest_benchmark is not installed, "
          "falling back to pytest_benchmark fixtures.\n")

    # if rapids_pytest_benchmark is not available, just perfrom time-only
    # benchmarking and replace the util functions with nops
    import pytest_benchmark
    gpubenchmark = pytest_benchmark.plugin.benchmark

    def setFixtureParamNames(*args, **kwargs):
        pass


###############################################################################
# Fixtures
@pytest.fixture(scope="module")
def client_connection():
    # setup
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize(p2p=True)

    yield client

    # teardown
    Comms.destroy()
    client.close()
    cluster.close()


@pytest.fixture(scope="module",
                params=utils.DATASETS_UNDIRECTED)
def daskGraphFromDataset(request, client_connection):
    """
    Returns a new dask dataframe created from the dataset file param.
    """
    # Since parameterized fixtures do not assign param names to param values,
    # manually call the helper to do so.
    setFixtureParamNames(request, ["dataset"])
    dataset = request.param

    chunksize = dcg.get_chunksize(dataset)
    ddf = dask_cudf.read_csv(dataset, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, 'src', 'dst')
    return dg


###############################################################################
# Tests
def test_mg_louvain_with_edgevals(daskGraphFromDataset):
    # FIXME: daskGraphFromDataset returns a DiGraph, which Louvain is currently
    # accepting. In the future, an MNMG symmeterize will need to be called to
    # create a Graph for Louvain.
    parts, mod = dcg.louvain(daskGraphFromDataset)

    # FIXME: either call Nx with the same dataset and compare results, or
    # hadcode golden results to compare to.
    print()
    print(parts.compute())
    print(mod)
    print()
