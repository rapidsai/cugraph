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

import time
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
    MultiGraph,
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
    G = MultiGraph(directed=True)

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
        enable_tcp_over_ucx=False,
        enable_infiniband=False,
        enable_nvlink=False,
        enable_rdmacm=False,
        net_devices=None,
    )
    # FIXME: need to consider directed/undirected?
    G = MultiGraph(directed=True)

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
    G, seed, batch_size, fanout, with_replacement
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
    if with_replacement not in [True, False]:
        raise ValueError(f"got unexpected value {with_replacement=}")

    rng = np.random.default_rng(seed)
    num_verts = G.number_of_vertices()

    if batch_size > num_verts:
        num_start_verts = int(num_verts * 0.25)
    else:
        num_start_verts = batch_size

    # Create the list of starting vertices by picking num_start_verts random
    # ints between 0 and num_verts, then map those to actual vertex IDs.  Since
    # the randomly-chosen IDs may not map to actual IDs, keep trying until
    # num_start_verts have been picked, or max_tries is reached.
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

    return {
        "start_list": list(start_list),
        "fanout": fanout,
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

    print("creating graph...")
    st = time.perf_counter_ns()
    if gpu_config == "SG":
        G = create_graph(graph_data)
        uns_func = uniform_neighbor_sample
    else:
        (G, dask_client, dask_cluster) = create_mg_graph(graph_data)
        uns_func = uniform_neighbor_sample_mg
        def uns_func(*args, **kwargs):
            print("running sampling...")
            st = time.perf_counter_ns()
            result_ddf = uniform_neighbor_sample_mg(*args, **kwargs)
            print(f"done running sampling, took {((time.perf_counter_ns() - st) / 1e9)}s")
            print("dask compute() results...")
            st = time.perf_counter_ns()
            sources = result_ddf["sources"].compute().to_cupy()
            destinations = result_ddf["destinations"].compute().to_cupy()
            indices = result_ddf["indices"].compute().to_cupy()
            print(f"done dask compute() results, took {((time.perf_counter_ns() - st) / 1e9)}s")
            return (sources, destinations, indices)

    print(f"done creating graph, took {((time.perf_counter_ns() - st) / 1e9)}s")

    yield (G, uns_func)

    if dask_client is not None:
        stop_dask_client(dask_client, dask_cluster)


################################################################################
# Benchmarks
@pytest.mark.parametrize("batch_size", params.batch_sizes.values())
@pytest.mark.parametrize("fanout", [params.fanout_10_25, params.fanout_5_10_15])
@pytest.mark.parametrize(
    "with_replacement", [False], ids=lambda v: f"with_replacement={v}"
)
def bench_cugraph_uniform_neighbor_sample(
    gpubenchmark, graph_objs, batch_size, fanout, with_replacement
):
    (G, uniform_neighbor_sample_func) = graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, _seed, batch_size, fanout, with_replacement
    )
    # print(f"\n{uns_args}")
    # FIXME: uniform_neighbor_sample cannot take a np.ndarray for start_list
    result = gpubenchmark(
        uniform_neighbor_sample_func,
        G,
        start_list=uns_args["start_list"],
        fanout_vals=uns_args["fanout"],
        with_replacement=uns_args["with_replacement"],
    )
    dtmap = {"int32": 32 // 8, "int64": 64 // 8}
    if isinstance(result, tuple):
        dt = str(result[0].dtype)
        llen = len(result[0])
    else:
        dt = str(result.sources.dtype)
        llen = len(result.sources)
    print(f"\nresult list len: {llen} (x3), dtype={dt}, total bytes={3*llen*dtmap[dt]}")
