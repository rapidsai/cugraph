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


# pytest param values used for defining input combinations. These also include
# markers for easily running specific combinations.
_sg_param_val = pytest.param("SG", marks=[pytest.mark.sg])
_snmg_param_val = pytest.param("SNMG", marks=[pytest.mark.snmg, pytest.mark.mg])
_mnmg_param_val = pytest.param("MNMG", marks=[pytest.mark.mnmg, pytest.mark.mg])
_graph_param_val = pytest.param(Graph, marks=[pytest.mark.local])
_remotegraph_param_val = pytest.param(RemoteGraph, marks=[pytest.mark.remote])

# Define/generate the combinations to run.
_graph_obj_fixture_params = gen_fixture_params(
    (_graph_param_val, _sg_param_val),
    (_graph_param_val, _snmg_param_val),
    (_remotegraph_param_val, _sg_param_val),
    (_remotegraph_param_val, _snmg_param_val),
    (_remotegraph_param_val, _mnmg_param_val),
    ids=lambda params: f"class={params[0].__name__},config={params[1]}",
)


@pytest.fixture(scope="module", params=_graph_obj_fixture_params)
def graph_obj(request):
    (graph_type, gpu_config) = request.param
    if graph_type is Graph:
        pass
    elif graph_type is RemoteGraph:
        pass
    else:
        raise RuntimeError(f"{graph_type} is invalid")

    G = None
    yield G


# @pytest.mark.parametrize("start_list__fanout_vals", [([0, 1, 2], [1, 2, 1])])
# def bench_uniform_neighbor_sample(gpubenchmark, graph_obj, start_list__fanout_vals):
def bench_uniform_neighbor_sample(gpubenchmark, graph_obj):
    """
    The graph_obj fixture is parameterized to be one of the following: "local
    SG", "local MG", "remote SG", "remote MG"
    """
    # (start_list, fanout_vals) = start_list__fanout_vals
    import time

    def f():
        time.sleep(0.1)

    gpubenchmark(f)
