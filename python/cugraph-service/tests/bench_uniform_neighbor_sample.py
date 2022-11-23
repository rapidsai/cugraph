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

from pathlib import Path

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

from cugraph_service_client import CugraphServiceClient
from cugraph_service_client.exceptions import CugraphServiceError
from cugraph_service_client import RemoteGraph
from . import utils

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


def create_graph(graph_data, gpu_config):
    """
    Create a graph instance based on the data to be loaded/generated and
    the GPU configuration (single-GPU, multi-GPU, etc.)
    """
    is_mg = gpu_config in ["SNMG", "MNMG"]
    # FIXME: need to consider directed/undirected?
    G = Graph()

    if isinstance(graph_data, datasets.Dataset):
        edgelist_df = graph_data.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        if is_mg:
            edgelist_df = dask_cudf.from_cudf(edgelist_df)
            G.from_dask_cudf_edgelist(
                edgelist_df, source="src", destination="dst", edge_attr="wgt"
            )
        else:
            G.from_cudf_edgelist(
                edgelist_df, source="src", destination="dst", edge_attr="wgt"
            )

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
            scramble_vertex_ids=False,  # FIXME: need to understand relevance of this
            create_using=None,  # None == return edgelist
            mg=is_mg,
        )
        rng = np.random.default_rng(seed)
        edgelist_df["weight"] = rng.random(size=len(edgelist_df))

        if is_mg:
            G.from_dask_cudf_edgelist(
                edgelist_df, source="src", destination="dst", edge_attr="weight"
            )
        else:
            G.from_cudf_edgelist(
                edgelist_df, source="src", destination="dst", edge_attr="weight"
            )

    else:
        raise TypeError(
            "graph_data can only be Dataset or dict, " f"got {type(graph_data)}"
        )

    return G


def create_remote_graph(graph_data, client):
    """
    Create a remote graph instance based on the data to be loaded/generated,
    relying on server-side graph creation extensions.
    """
    if isinstance(graph_data, datasets.Dataset):
        # breakpoint()
        gid = client.call_graph_creation_extension(
            "create_graph_from_dataset", graph_data.meta_data["name"]
        )

    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        gid = client.call_graph_creation_extension("create_graph_from_rmat", graph_data)

    else:
        raise TypeError(
            "graph_data can only be Dataset or dict, " f"got {type(graph_data)}"
        )

    G = RemoteGraph(client, gid)
    return G


def get_uniform_neighbor_sample_args(
    G, seed, start_list_len, fanout_list_len, with_replacement
):
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
    if with_replacement not in [True, False]:
        raise ValueError(f"got unexpected value {with_replacement=}")

    rng = np.random.default_rng(seed)
    num_verts = G.number_of_vertices()

    if start_list_len == "LARGE":
        num_start_verts = min(1000, int(num_verts * 0.25))
    else:
        num_start_verts = 2

    # start_list = rng.choice(num_verts, min(1000, int(num_verts * 0.25)))
    # FIXME: need a better way to get unique starting verts
    starts = set()
    edgelist = G.edges()
    for _ in range(num_start_verts):
        i = rng.choice(num_verts, 1)[0]
        # FIXME: must be "src" even though input edgelist used "source"
        col = "src"
        v = edgelist[col].loc[i]
        while v in starts:
            i = rng.choice(num_verts, 1)[0]
            v = edgelist[col].loc[i]
        starts.add(v)
    start_list = list(starts)

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
        "with_replacement": with_replacement,
    }


def ensure_running_service():
    """
    Returns a tuple containing a Popen object for the running cugraph-service
    server subprocess, and a client object connected to it.  If a server was
    detected already running, the Popen object will be None.
    """
    host = "localhost"
    port = 9090
    client = CugraphServiceClient(host, port)
    server_process = None

    try:
        client.uptime()
        print("FOUND RUNNING SERVER, ASSUMING IT SHOULD BE USED FOR TESTING!")

    except CugraphServiceError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.
        server_process = utils.start_server_subprocess(host=host, port=port)

    # Ensure the extensions needed for these benchmarks are loaded
    required_graph_creation_extension_module = "benchmark_server_extension"
    server_data = client.get_server_info()
    # .stem excludes .py extensions, so it can match a python module name
    loaded_graph_creation_extension_modules = [
        Path(m).stem for m in server_data["graph_creation_extensions"]
    ]
    if (
        required_graph_creation_extension_module
        not in loaded_graph_creation_extension_modules
    ):
        modules_loaded = client.load_graph_creation_extensions(
            "cugraph_service_server.testing.benchmark_server_extension"
        )
        if len(modules_loaded) < 1:
            raise RuntimeError(
                "failed to load graph creation extension "
                f"{required_graph_creation_extension_module}"
            )

    return (server_process, client)


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
    server_process = None

    if gpu_config not in ["SG", "SNMG", "MNMG"]:
        raise RuntimeError(f"got unexpected gpu_config value: {gpu_config}")

    if graph_type is Graph:
        G = create_graph(graph_data, gpu_config)
        if gpu_config == "SG":
            uns_func = uniform_neighbor_sample
        else:
            uns_func = uniform_neighbor_sample_mg

    elif graph_type is RemoteGraph:
        # Ensure a server is running
        if gpu_config == "SG":
            (server_process, client) = ensure_running_service()

        elif gpu_config == "SNMG":
            pass
        else:
            pass

        G = create_remote_graph(graph_data, client)

    else:
        raise RuntimeError(f"{graph_type=} is invalid")

    yield (G, uns_func)

    del G  # is this necessary?
    if server_process is not None:
        print("\nTerminating server...", end="", flush=True)
        server_process.terminate()
        server_process.wait(timeout=60)
        print("done.", flush=True)


################################################################################
# Benchmarks
@pytest.mark.parametrize("start_list_len", [small_start_list_pv, large_start_list_pv])
@pytest.mark.parametrize(
    "fanout_vals_len", [small_fanout_list_pv, large_fanout_list_pv]
)
@pytest.mark.parametrize(
    "with_replacement", [True, False], ids=lambda v: f"with_replacement={v}"
)
def bench_uniform_neighbor_sample(
    gpubenchmark, graph_objs, start_list_len, fanout_vals_len, with_replacement
):
    """
    @pytest.mark.parametrize("start_list_len",
                             [small_start_list_pv, large_start_list_pv])
    @pytest.mark.parametrize(
        "fanout_vals_len", [small_fanout_list_pv, large_fanout_list_pv]
    )
    @pytest.mark.parametrize("with_replacement", [True, False])
    def bench_uniform_neighbor_sample(
        graph_objs, start_list_len, fanout_vals_len, with_replacement):
    """
    (G, uniform_neighbor_sample_func) = graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, seed, start_list_len, fanout_vals_len, with_replacement
    )
    # print(f"\n{uns_args}")
    # FIXME: uniform_neighbor_sample cannot take a np.ndarray for start_list
    gpubenchmark(
        uniform_neighbor_sample_func,
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
    """
