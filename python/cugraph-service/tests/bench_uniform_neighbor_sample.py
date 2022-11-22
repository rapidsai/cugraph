# Copyright (c) 2022, NVIDIA CORPORATION.
#
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

"""
uniform neighbor sample

cugraph ("local")
  SG - small graph
    small sample, large sample
  SNMG - medium graph
    small sample, large sample
  MNMG - large graph
    small sample, large sample

cugraph-service
  SG - small graph
    small sample, large sample
  SNMG - medium graph
    small sample, large sample
  MNMG - large graph
    small sample, large sample
"""

import pytest
from pylibcugraph.testing.utils import gen_fixture_params

# If the rapids-pytest-benchmark plugin is installed, the "gpubenchmark"
# fixture will be available automatically. Check that this fixture is available
# by trying to import rapids_pytest_benchmark, and if that fails, set
# "gpubenchmark" to the standard "benchmark" fixture provided by
# pytest-benchmark.
try:
    import rapids_pytest_benchmark  # noqa: F401
except ImportError:
    import pytest_benchmark

    gpubenchmark = pytest_benchmark.plugin.benchmark

from cugraph import Graph
from cugraph_service_client import RemoteGraph
from cugraph.experimental import datasets

# from cugraph.generators import rmat

# pytest param values ("pv" suffix) used for defining input combinations. These
# also include markers for easily running specific combinations.
sg_pv = pytest.param(
    "SG",
    marks=[pytest.mark.sg],
    id="config=SG",
)
snmg_pv = pytest.param(
    "SNMG",
    marks=[pytest.mark.snmg, pytest.mark.mg],
    id="config=SNMG",
)
mnmg_pv = pytest.param(
    "MNMG",
    marks=[pytest.mark.mnmg, pytest.mark.mg],
    id="config=MNMG",
)
graph_pv = pytest.param(
    Graph,
    marks=[pytest.mark.local],
    id="type=Graph",
)
remotegraph_pv = pytest.param(
    RemoteGraph,
    marks=[pytest.mark.remote],
    id="type=RemoteGraph",
)
karate_pv = pytest.param(
    datasets.karate,
    id="dataset=karate",
)
small_low_degree_rmat_pv = pytest.param(
    {"scale": 16, "edgefactor": 4, "seed": 42},
    id="dataset=rmat_16_4",
)
small_high_degree_rmat_pv = pytest.param(
    {"scale": 16, "edgefactor": 32, "seed": 42},
    id="dataset=rmat_16_32",
)
large_low_degree_rmat_pv = pytest.param(
    {"scale": 24, "edgefactor": 4, "seed": 42},
    id="dataset=rmat_24_4",
)
large_high_degree_rmat_pv = pytest.param(
    {"scale": 24, "edgefactor": 32, "seed": 42},
    id="dataset=rmat_24_32",
)
huge_low_degree_rmat_pv = pytest.param(
    {"scale": 30, "edgefactor": 4, "seed": 42},
    id="dataset=rmat_30_4",
)
huge_high_degree_rmat_pv = pytest.param(
    {"scale": 30, "edgefactor": 32, "seed": 42},
    id="dataset=rmat_30_32",
)
large_start_list_pv = pytest.param(
    "LARGE",
    marks=[pytest.mark.start_list_large],
    id="start_list_len=LARGE",
)
small_start_list_pv = pytest.param(
    "SMALL",
    marks=[pytest.mark.start_list_small],
    id="start_list_len=SMALL",
)
large_fanout_list_pv = pytest.param(
    "LARGE",
    marks=[pytest.mark.fanout_list_large],
    id="fanout_list_len=LARGE",
)
small_fanout_list_pv = pytest.param(
    "SMALL",
    marks=[pytest.mark.fanout_list_small],
    id="fanout_list_len=SMALL",
)
# Define/generate the combinations to run.
graph_obj_fixture_params = gen_fixture_params(
    (graph_pv, sg_pv, karate_pv),
    (graph_pv, sg_pv, small_low_degree_rmat_pv),
    (graph_pv, sg_pv, small_high_degree_rmat_pv),
    (graph_pv, snmg_pv, large_low_degree_rmat_pv),
    (graph_pv, snmg_pv, large_high_degree_rmat_pv),
    (remotegraph_pv, sg_pv, karate_pv),
    (remotegraph_pv, sg_pv, small_low_degree_rmat_pv),
    (remotegraph_pv, sg_pv, small_high_degree_rmat_pv),
    (remotegraph_pv, snmg_pv, large_low_degree_rmat_pv),
    (remotegraph_pv, snmg_pv, large_high_degree_rmat_pv),
    (remotegraph_pv, mnmg_pv, huge_low_degree_rmat_pv),
    (remotegraph_pv, mnmg_pv, huge_high_degree_rmat_pv),
)


@pytest.fixture(scope="module", params=graph_obj_fixture_params)
def graph_obj(request):
    """
    (graph_type, gpu_config) = request.param
    if graph_type is Graph:
        G = cugraph.Graph()
        if config == "SG":
            # G.from_cudf_edgelist()
        elif config == "SNMG":
            # G.from_dask_cudf_edgelist()
        else:
            raise RuntimeError(f"got unexpected config value for Graph: {config}")

    elif graph_type is RemoteGraph:
        if config == "SG":
            client = create_sg_client()
            G = RemoteGraph()


    else:
        raise RuntimeError(f"{graph_type} is invalid")
    """
    G = None
    yield G


@pytest.mark.parametrize("start_list_len", [small_start_list_pv, large_start_list_pv])
@pytest.mark.parametrize(
    "fanout_vals_len", [small_fanout_list_pv, large_fanout_list_pv]
)
def bench_uniform_neighbor_sample(
    gpubenchmark, graph_obj, start_list_len, fanout_vals_len
):
    """
    The graph_obj fixture is parameterized to be one of the following: "local
    SG", "local MG", "remote SG", "remote MG"
    """
    # (start_list, fanout_vals) = start_list__fanout_vals
    import time

    def f():
        time.sleep(0.01)

    gpubenchmark(f)
