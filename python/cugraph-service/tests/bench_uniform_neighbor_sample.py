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
import numpy as np
from pylibcugraph.testing.utils import gen_fixture_params
import dask_cudf

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
from cugraph.experimental import datasets
from cugraph.generators import rmat

from cugraph_service_client import RemoteGraph

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


def get_edgelist(graph_data, gpu_config):
    is_mg = gpu_config in ["SNMG", "MNMG"]

    if isinstance(graph_data, datasets.Dataset):
        edgelist_df = graph_data.get_edgelist()
        if is_mg:
            return dask_cudf.from_cudf(edgelist_df)
        else:
            return edgelist_df
    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        scale = graph_data["scale"]
        num_edges = (2**scale) * graph_data["edgefactor"]
        seed = graph_data["seed"]
        edgelist_df = rmat(
            scale,
            num_edges,
            0.57,  # from Graph500
            0.19,  # from Graph500
            0.19,  # from Graph500
            seed,
            clip_and_flip=False,
            scramble_vertex_ids=True,
            create_using=None,  # None == return edgelist
            mg=is_mg,
        )
        rng = np.random.default_rng(seed)
        edgelist_df["weight"] = rng.random(size=len(edgelist_df))
        return edgelist_df
    else:
        raise TypeError(
            "graph_data can only be Dataset or dict, " f"got {type(graph_data)}"
        )


@pytest.fixture(scope="module", params=graph_obj_fixture_params)
def graph_obj(request):
    (graph_type, gpu_config, graph_data) = request.param
    if graph_type is Graph:
        G = Graph()
        edgelist_df = get_edgelist(graph_data, gpu_config)
        if gpu_config == "SG":
            G.from_cudf_edgelist(edgelist_df)
        elif gpu_config == "SNMG":
            G.from_dask_cudf_edgelist(edgelist_df)
        elif gpu_config == "MNMG":
            raise NotImplementedError("config=MNMG")
        else:
            raise RuntimeError(
                f"got unexpected gpu_config value for Graph: {gpu_config}"
            )

    elif graph_type is RemoteGraph:
        raise NotImplementedError
        if gpu_config == "SG":
            # client = create_sg_client()
            G = RemoteGraph()

    else:
        raise RuntimeError(f"{graph_type} is invalid")

    yield G


@pytest.mark.parametrize("start_list_len", [small_start_list_pv, large_start_list_pv])
@pytest.mark.parametrize(
    "fanout_vals_len", [small_fanout_list_pv, large_fanout_list_pv]
)
def bench_uniform_neighbor_sample(
    gpubenchmark, graph_obj, start_list_len, fanout_vals_len
):
    """ """
    # G = graph_obj
    # breakpoint()
    import time

    def f():
        time.sleep(0.01)

    gpubenchmark(f)
