# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import os

import pytest
import numpy as np
import cupy as cp
from cugraph.testing.mg_utils import start_dask_client, stop_dask_client
import cudf
import dask_cudf
import rmm

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
from cugraph_benchmarking.timer import TimerContext

_seed = 42


def create_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated, return a
    tuple containing (graph_obj, num_verts)
    """
    # FIXME: need to consider directed/undirected?
    G = MultiGraph(directed=True)

    rmm.reinitialize(pool_allocator=True)

    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        G.from_cudf_edgelist(
            edgelist_df, source="src", destination="dst", edge_attr="wgt", legacy_renum_only=True
        )
        num_verts = G.number_of_vertices()

    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        scale = graph_data["scale"]
        num_verts = 2**scale
        num_edges = num_verts * graph_data["edgefactor"]
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
            legacy_renum_only=True,
        )

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return (G, num_verts)


def create_mg_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated, return a
    tuple containing (graph_obj, num_verts, client, cluster)
    """
    n_devices = os.getenv("DASK_NUM_WORKERS", 4)
    n_devices = int(n_devices)
    # range starts at 1 to let let 0 be used by benchmark/client process
    visible_devices = ",".join([str(i) for i in range(1, n_devices+1)])

    (client, cluster) = start_dask_client(
        # enable_tcp_over_ucx=True,
        # enable_infiniband=False,
        # enable_nvlink=True,
        # enable_rdmacm=False,
        protocol="ucx",
        rmm_pool_size="28GB",
        dask_worker_devices=visible_devices,
    )
    rmm.reinitialize(pool_allocator=True)

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
            edgelist_df, source="src", destination="dst", edge_attr="wgt", legacy_renum_only=True
        )
        num_verts = G.number_of_vertices()

    # Assume dictionary contains RMAT params
    elif isinstance(graph_data, dict):
        scale = graph_data["scale"]
        num_verts = 2**scale
        num_edges = num_verts * graph_data["edgefactor"]
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
            legacy_renum_only=True,
        )

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return (G, num_verts, client, cluster)


def get_uniform_neighbor_sample_args(
    G, num_verts, seed, batch_size, fanout, with_replacement
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

    # start_list is a random sampling of the src verts.
    # Dask series only support the frac arg for getting n samples.
    num_start_verts = batch_size
    srcs = G.edgelist.edgelist_df["src"]
    frac = num_start_verts / num_verts
    start_list = srcs.sample(frac=frac, random_state=seed)

    # Attempt to automatically handle a dask Series
    if hasattr(start_list, "compute"):
        start_list = start_list.compute()

    # frac does not guarantee exactly num_start_verts, so ensure only
    # num_start_verts are returned
    start_list = start_list[:num_start_verts]
    assert len(start_list) == num_start_verts
    start_list = start_list.to_numpy().tolist()

    return {
        "start_list": start_list,
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

    with TimerContext("creating graph"):
        if gpu_config == "SG":
            (G, num_verts) = create_graph(graph_data)
            uns_func = uniform_neighbor_sample
        else:
            (G, num_verts, dask_client, dask_cluster) = create_mg_graph(graph_data)
            # The default uniform_neighbor_sample MG function returns a
            # dask_cudf DataFrame, which must be evaluated (.compute()) in
            # order to get usable results.
            # Uncomment to benchmark with a dask_cudf DataFrame
            #
            # uns_func = uniform_neighbor_sample_mg

            # This function calls the MG uniform_neighbor_sample function, then
            # calls .compute() on the resulting dask_cudf DataFrame in order to
            # return usable results.
            # Uncomment this to benchmark with results available to client
            #
            def uns_func(*args, **kwargs):
                with TimerContext("sampling"):
                    result_ddf = uniform_neighbor_sample_mg(*args, **kwargs)

                with TimerContext("dask compute() results"):
                    result_df = result_ddf.compute()
                    sources = result_df["sources"].to_cupy()
                    destinations = result_df["destinations"].to_cupy()
                    indices = result_df["indices"].to_cupy()

                return (sources, destinations, indices)

    yield (G, num_verts, uns_func)

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
    (G, num_verts, uniform_neighbor_sample_func) = graph_objs

    with TimerContext("computing sampling args"):
        uns_args = get_uniform_neighbor_sample_args(
            G, num_verts, _seed, batch_size, fanout, with_replacement
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
    """
    dtmap = {"int32": 32 // 8, "int64": 64 // 8}
    if isinstance(result, tuple):
        dt = str(result[0].dtype)
        llen = len(result[0])
    else:
        dt = str(result.sources.dtype)
        llen = len(result.sources)
    print(f"\nresult list len: {llen} (x3), dtype={dt}, total bytes={3*llen*dtmap[dt]}")
    """
