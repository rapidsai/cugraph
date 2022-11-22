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

from cugraph import (
    Graph,
    uniform_neighbor_sample,
)
from cugraph.generators import rmat
from cugraph.experimental import datasets
from cugraph.dask import uniform_neighbor_sample as uniform_neighbor_sample_mg

from cugraph_service_client import RemoteGraph

seed = 42

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
    {"scale": 16, "edgefactor": 4, "seed": seed},
    id="dataset=rmat_16_4",
)
small_high_degree_rmat_pv = pytest.param(
    {"scale": 16, "edgefactor": 32, "seed": seed},
    id="dataset=rmat_16_32",
)
large_low_degree_rmat_pv = pytest.param(
    {"scale": 24, "edgefactor": 4, "seed": seed},
    id="dataset=rmat_24_4",
)
large_high_degree_rmat_pv = pytest.param(
    {"scale": 24, "edgefactor": 32, "seed": seed},
    id="dataset=rmat_24_32",
)
huge_low_degree_rmat_pv = pytest.param(
    {"scale": 30, "edgefactor": 4, "seed": seed},
    id="dataset=rmat_30_4",
)
huge_high_degree_rmat_pv = pytest.param(
    {"scale": 30, "edgefactor": 32, "seed": seed},
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
    """
    Return an edgelist DataFrame based on the data to be loaded/generated and
    the GPU configuration (single-GPU, multi-GPU, etc.)
    """
    is_mg = gpu_config in ["SNMG", "MNMG"]

    if isinstance(graph_data, datasets.Dataset):
        edgelist_df = graph_data.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        edgelist_df.rename(
            columns={"src": "source", "dst": "destination", "wgt": "weight"},
            inplace=True,
        )
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
        edgelist_df.rename(
            columns={"src": "source", "dst": "destination"}, inplace=True
        )
        return edgelist_df

    else:
        raise TypeError(
            "graph_data can only be Dataset or dict, " f"got {type(graph_data)}"
        )


def get_uniform_neighbor_sample_args(G, seed, start_list_len, fanout_list_len):
    """
    Return a dictionary containing the args for uniform_neighbor_sample based
    on the graph and desired args passed in. For example, if a large start list
    and small fanout list is desired, a "large" (based on graph size) list of
    valid vert IDs for the graph passed in and a "small" list of fanout values
    will be returned.

    The dictionary return value allows for easily supporting other args without
    having to maintain an order of values in a return tuple, for example.
    """
    if start_list_len not in ["SMALL", "LARGE"]:
        raise ValueError(f"got unexpected value {start_list_len=}")
    if fanout_list_len not in ["SMALL", "LARGE"]:
        raise ValueError(f"got unexpected value {fanout_list_len=}")

    rng = np.random.default_rng(seed)
    num_verts = G.number_of_vertices()

    if start_list_len == "LARGE":
        start_list = rng.choice(num_verts, min(1000, int(num_verts * 0.5)))
    else:
        start_list = rng.choice(num_verts, 2)

    if fanout_list_len == "LARGE":
        num_edges = G.number_of_edges()
        avg_degree = num_edges // num_verts
        max_fanout = min(avg_degree, 5)
        if max_fanout == 1:
            fanout_choices = [1]
        else:
            fanout_choices = np.arange(1, max_fanout)
        fanout_list = [rng.choice(fanout_choices, 1)[0] for _ in range(5)]
    else:
        fanout_list = [2, 1]

    return {
        "start_list": list(start_list),
        "fanout_vals": list(fanout_list),
        "with_replacement": True,
    }


@pytest.fixture(scope="module", params=graph_obj_fixture_params)
def graph_objs(request):
    """
    Fixture that returns a Graph object and algo callable based on the
    parameters. This handles instantiating the correct type (Graph or
    RemoteGraph) and populating it with graph data. This also ensures a
    cugraph-service server is running in the case of a RemoteGraph. The
    callable returned will be appropriate for the graph type (ie. "local" or
    "remote" algo API).
    """
    (graph_type, gpu_config, graph_data) = request.param

    if gpu_config not in ["SG", "SNMG", "MNMG"]:
        raise RuntimeError(f"got unexpected gpu_config value: {gpu_config}")

    if graph_type is Graph:
        # FIXME: need to consider directed/undirected?
        G = Graph()
        edgelist_df = get_edgelist(graph_data, gpu_config)
        if gpu_config == "SG":
            G.from_cudf_edgelist(edgelist_df)
            uns_func = uniform_neighbor_sample
        else:
            G.from_dask_cudf_edgelist(edgelist_df)
            uns_func = uniform_neighbor_sample_mg

    elif graph_type is RemoteGraph:
        raise NotImplementedError
        if gpu_config == "SG":
            # Ensure SG server is running
            # client = create_sg_client()
            G = RemoteGraph()
        elif gpu_config == "SNMG":
            # Ensure SNMG server is running
            pass
        else:
            # Ensure MNMG server is running
            pass

    else:
        raise RuntimeError(f"{graph_type=} is invalid")

    yield (G, uns_func)


################################################################################
# Benchmarks
"""
@pytest.mark.parametrize("start_list_len", [small_start_list_pv, large_start_list_pv])
@pytest.mark.parametrize(
    "fanout_vals_len", [small_fanout_list_pv, large_fanout_list_pv]
)
def bench_uniform_neighbor_sample(
    gpubenchmark, graph_objs, start_list_len, fanout_vals_len
):
"""


@pytest.mark.parametrize("start_list_len", [small_start_list_pv, large_start_list_pv])
@pytest.mark.parametrize(
    "fanout_vals_len", [small_fanout_list_pv, large_fanout_list_pv]
)
def bench_uniform_neighbor_sample(graph_objs, start_list_len, fanout_vals_len):
    """ """
    (G, uniform_neighbor_sample_func) = graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, seed, start_list_len, fanout_vals_len
    )
    print(f"\n{uns_args}")
    # FIXME: uniform_neighbor_sample cannot take a np.ndarray for start_list
    """
    gpubenchmark(uniform_neighbor_sample_func,
                 G,
                 start_list=uns_args["start_list"],
                 fanout_vals=uns_args["fanout_vals"],
                 with_replacement=uns_args["with_replacement"],
                 )
    """
    uniform_neighbor_sample_func(
        G,
        start_list=uns_args["start_list"],
        fanout_vals=uns_args["fanout_vals"],
        with_replacement=uns_args["with_replacement"],
    )
