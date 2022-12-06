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
import cupy as cp
from cugraph.testing.mg_utils import start_dask_client, stop_dask_client
import cudf
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

from cugraph_benchmarking import params

_seed = 42


def create_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
    # FIXME: need to consider directed/undirected?
    G = Graph()

    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        G.from_cudf_edgelist(
            edgelist_df, source="src", destination="dst", edge_attr="wgt", renumber=True
        )

    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        scale = graph_data["scale"]
        num_edges = (2**scale) * graph_data["edgefactor"]
        seed = _seed
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
            mg=False,
        )
        edgelist_df["weight"] = cp.float32(1)

        G.from_cudf_edgelist(
            edgelist_df,
            source="src",
            destination="dst",
            edge_attr="weight",
            renumber=True,
        )

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return G


def create_mg_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
    (client, cluster) = start_dask_client(
        enable_tcp_over_ucx=True,
        enable_nvlink=True,
        enable_infiniband=False,
        enable_rdmacm=False,
        net_devices=None,
    )
    # FIXME: need to consider directed/undirected?
    G = Graph()

    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        edgelist_df = dask_cudf.from_cudf(edgelist_df)
        G.from_dask_cudf_edgelist(
            edgelist_df, source="src", destination="dst", edge_attr="wgt", renumber=True
        )

    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        scale = graph_data["scale"]
        num_edges = (2**scale) * graph_data["edgefactor"]
        seed = _seed
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
            mg=True,
        )
        edgelist_df["weight"] = np.float32(1)

        G.from_dask_cudf_edgelist(
            edgelist_df,
            source="src",
            destination="dst",
            edge_attr="weight",
            renumber=True,
        )

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return (G, client, cluster)


def get_uniform_neighbor_sample_args(
    G, seed, start_list_len, fanout, with_replacement
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
    if fanout not in ["SMALL", "LARGE"]:
        raise ValueError(f"got unexpected value {fanout=}")
    if with_replacement not in [True, False]:
        raise ValueError(f"got unexpected value {with_replacement=}")

    rng = np.random.default_rng(seed)
    num_verts = G.number_of_vertices()
    num_edges = G.number_of_edges()

    if start_list_len > num_verts:
        num_start_verts = int(num_verts * 0.25)
    else:
        num_start_verts = start_list_len

    assert G.renumbered
    start_list_set = set()
    max_tries = 10000
    try_num = 0
    while (len(start_list_set) < num_start_verts) and (try_num < max_tries):
        internal_vertex_ids_start_list = rng.choice(
            num_verts, size=num_start_verts, replace=False
        )
        start_list_df = cudf.DataFrame({"vid": internal_vertex_ids_start_list})
        start_list_df = G.unrenumber(start_list_df, "vid")

        if G.is_multi_gpu():
            start_list_series = start_list_df.compute()["vid"]
        else:
            start_list_series = start_list_df["vid"]

        start_list_series.dropna(inplace=True)
        start_list_set.update(set(start_list_series.values_host.tolist()))
        try_num += 1

    start_list = list(start_list_set)
    start_list = start_list[:num_start_verts]
    assert len(start_list) == num_start_verts

    # Generate a fanout list based on degree if the list is to be large,
    # otherwise just use a small list of fixed numbers.
    if fanout == "LARGE":
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


@pytest.fixture(scope="module", params=params.graph_obj_fixture_params)
def graph_objs(request):
    """
    Fixture that returns a Graph object and algo callable (SG or MG) based on
    the parameters. This handles instantiating the correct type (SG or MG) and
    populating it with graph data.
    """
    (gpu_config, graph_data) = request.param
    dask_client = None
    dask_cluster = None

    if gpu_config not in ["SG", "SNMG", "MNMG"]:
        raise RuntimeError(f"got unexpected gpu_config value: {gpu_config}")

    if gpu_config == "SG":
        G = create_graph(graph_data)
        uns_func = uniform_neighbor_sample
    else:
        (G, dask_client, dask_cluster) = create_mg_graph(graph_data)
        uns_func = uniform_neighbor_sample_mg

    yield (G, uns_func)

    if dask_client is not None:
        stop_dask_client(dask_client, dask_cluster)


################################################################################
# Benchmarks
@pytest.mark.parametrize("start_list_len", params.start_list.values())
@pytest.mark.parametrize("fanout", [params.fanout_small, params.fanout_large])
@pytest.mark.parametrize(
    "with_replacement", [False], ids=lambda v: f"with_replacement={v}"
)
def bench_cugraph_uniform_neighbor_sample(
    gpubenchmark, graph_objs, start_list_len, fanout, with_replacement
):
    (G, uniform_neighbor_sample_func) = graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, _seed, start_list_len, fanout, with_replacement
    )
    # print(f"\n{uns_args}")
    # FIXME: uniform_neighbor_sample cannot take a np.ndarray for start_list
    result = gpubenchmark(
        uniform_neighbor_sample_func,
        G,
        start_list=uns_args["start_list"],
        fanout_vals=uns_args["fanout_vals"],
        with_replacement=uns_args["with_replacement"],
    )
    dtmap = {"int32": 32 // 8, "int64": 64 // 8}
    dt = str(result.sources.dtype)
    llen = len(result.sources)
    print(f"\nresult list len: {llen} (x3), dtype={dt}, total bytes={3*llen*dtmap[dt]}")
