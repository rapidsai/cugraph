# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import rmm
import numpy as np
from time import perf_counter_ns
from cugraph.dask.comms import comms as Comms
from cugraph.dask import uniform_neighbor_sample as uniform_neighbor_sample_mg
from cugraph import MultiGraph
from cugraph.generators import rmat

_seed = 42


def benchmark_func(func, n_times=10):
    def wrap_func(*args, **kwargs):
        time_ls = []
        # ignore 1st run
        # and return other runs
        for _ in range(0, n_times + 1):
            t1 = perf_counter_ns()
            result = func(*args, **kwargs)
            t2 = perf_counter_ns()
            time_ls.append(t2 - t1)
        return result, time_ls[1:]

    return wrap_func


def create_mg_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
    G = MultiGraph(directed=True)
    # Assume strings are names of datasets in the datasets package
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
    )
    return G


@benchmark_func
def sample_graph(G, start_list):
    output_ddf = uniform_neighbor_sample_mg(
        G, start_list=start_list, fanout_vals=[10, 25]
    )
    df = output_ddf.compute()
    return df


def run_sampling_test(ddf, start_list):
    df, time_ls = sample_graph(ddf, start_list)
    time_ar = np.asarray(time_ls)
    time_mean = time_ar.mean()
    print(f"Sampling {len(start_list):,} took = {time_mean * 1e-6} ms")
    return


if __name__ == "__main__":
    import os

    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

    cluster = LocalCUDACluster(
        protocol="ucx", rmm_pool_size="15GB", CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES
    )
    client = Client(cluster)
    Comms.initialize(p2p=True)

    rmm.reinitialize(pool_allocator=True)

    graph_data = {
        "scale": 26,
        "edgefactor": 8,
    }

    g = create_mg_graph(graph_data)

    for num_start_verts in [1_000, 10_000, 100_000]:
        start_list = g.input_df["src"].head(num_start_verts)
        assert len(start_list) == num_start_verts
        run_sampling_test(g, start_list)

    print("--" * 20 + "Completed Test" + "--" * 20, flush=True)

    Comms.destroy()
    client.shutdown()
    cluster.close()
