# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from cugraph.testing.mg_utils import (
    generate_edgelist_rmat,
    get_allocation_counts_dask_persist,
    get_allocation_counts_dask_lazy,
    sizeof_fmt,
    get_peak_output_ratio_across_workers,
    restart_client,
)

from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
    enable_spilling,
)
from cugraph.structure.symmetrize import symmetrize_ddf
import cugraph
import cudf
from time import sleep
import pandas as pd
import time


@get_allocation_counts_dask_lazy(return_allocations=True, logging=True)
def construct_graph(dask_dataframe, directed=False, renumber=False):
    """
    Args:
        dask_dataframe:
            dask_dataframe contains weighted and undirected edges with self
            loops. Multiple edges will likely be present as well.
        directed:
            If True, the graph will be directed.
        renumber:
            If True, the graph will be renumbered.
    Returns:
        G:  cugraph.Graph
    """
    st = time.time()
    G = cugraph.Graph(directed=directed)
    G.from_dask_cudf_edgelist(
        dask_dataframe, source="src", destination="dst", renumber=renumber
    )
    et = time.time()
    g_creation_time = et - st
    print(f"Graph creation time = {g_creation_time} s")
    return G, g_creation_time


@get_allocation_counts_dask_persist(return_allocations=True, logging=True)
def symmetrize_cugraph_df(dask_df, multi=False):
    output_df = symmetrize_ddf(dask_df, "src", "dst", multi=multi)
    return output_df


def benchmark_cugraph_graph_symmetrize(scale, edgefactor, seed, multi):
    """
    Benchmark cugraph graph symmetrization
    """
    dask_df = generate_edgelist_rmat(
        scale=scale, edgefactor=edgefactor, seed=seed, unweighted=True, mg=True
    )
    dask_df = dask_df.astype("int64")
    dask_df = dask_df.reset_index(drop=True)
    input_memory = dask_df.memory_usage().sum().compute()
    num_input_edges = len(dask_df)
    print(f"Number of input edges = {num_input_edges:,}, multi = {multi}")
    output_df, allocation_counts = symmetrize_cugraph_df(dask_df, multi=multi)
    (
        input_to_peak_ratio,
        output_to_peak_ratio,
        input_memory_per_worker,
        peak_allocation_across_workers,
    ) = get_memory_statistics(
        allocation_counts=allocation_counts, input_memory=input_memory
    )
    print(f"Number of edges after symmetrization = {len(output_df):,}")
    print("-" * 80)
    return (
        num_input_edges,
        input_to_peak_ratio,
        output_to_peak_ratio,
        input_memory_per_worker,
        peak_allocation_across_workers,
    )


def benchmark_cugraph_graph_creation(scale, edgefactor, seed, directed, renumber):
    """
    Entry point for the benchmark.
    """
    dask_df = generate_edgelist_rmat(
        scale=scale,
        edgefactor=edgefactor,
        seed=seed,
        unweighted=True,
        mg=True,
    )
    # We do below to remove the rmat memory overhead
    # which holds on to GPU memory
    dask_df = dask_df.map_partitions(lambda df: df.to_pandas()).persist()
    dask_df = dask_df.map_partitions(cudf.from_pandas)
    dask_df = dask_df.astype("int64")
    dask_df = dask_df.reset_index(drop=True)
    input_memory = dask_df.memory_usage().sum().compute()
    num_input_edges = len(dask_df)
    print(
        f"Number of input edges = {num_input_edges:,}, directed = {directed}, renumber = {renumber}"
    )
    (G, g_creation_time), allocation_counts = construct_graph(
        dask_df, directed=directed, renumber=renumber
    )
    (
        input_to_peak_ratio,
        output_to_peak_ratio,
        input_memory_per_worker,
        peak_allocation_across_workers,
    ) = get_memory_statistics(
        allocation_counts=allocation_counts, input_memory=input_memory
    )
    print(f"Number of edges in final graph = {G.number_of_edges():,}")
    print("-" * 80)
    return (
        num_input_edges,
        input_to_peak_ratio,
        output_to_peak_ratio,
        input_memory_per_worker,
        peak_allocation_across_workers,
        g_creation_time,
    )


def get_memory_statistics(allocation_counts, input_memory):
    """
    Get memory statistics for the benchmark.
    """
    output_to_peak_ratio = get_peak_output_ratio_across_workers(allocation_counts)
    peak_allocation_across_workers = max(
        [a["peak_bytes"] for a in allocation_counts.values()]
    )
    input_memory_per_worker = input_memory / len(allocation_counts.keys())
    input_to_peak_ratio = peak_allocation_across_workers / input_memory_per_worker
    print(f"Edge List Memory = {sizeof_fmt(input_memory_per_worker)}")
    print(f"Peak Memory across workers = {sizeof_fmt(peak_allocation_across_workers)}")
    print(f"Max Peak to output graph ratio across workers = {output_to_peak_ratio:.2f}")
    print(
        f"Max Peak to avg input graph ratio across workers = {input_to_peak_ratio:.2f}"
    )
    return (
        input_to_peak_ratio,
        output_to_peak_ratio,
        input_memory_per_worker,
        peak_allocation_across_workers,
    )


if __name__ == "__main__":
    client, cluster = start_dask_client(dask_worker_devices=[1], jit_unspill=False)
    enable_spilling()
    stats_ls = []
    client.run(enable_spilling)
    for scale in [23, 24, 25]:
        for directed in [True, False]:
            for renumber in [True, False]:
                try:
                    stats_d = {}
                    (
                        num_input_edges,
                        input_to_peak_ratio,
                        output_to_peak_ratio,
                        input_memory_per_worker,
                        peak_allocation_across_workers,
                        g_creation_time,
                    ) = benchmark_cugraph_graph_creation(
                        scale=scale,
                        edgefactor=16,
                        seed=123,
                        directed=directed,
                        renumber=renumber,
                    )
                    stats_d["scale"] = scale
                    stats_d["num_input_edges"] = num_input_edges
                    stats_d["directed"] = directed
                    stats_d["renumber"] = renumber
                    stats_d["input_memory_per_worker"] = sizeof_fmt(
                        input_memory_per_worker
                    )
                    stats_d["peak_allocation_across_workers"] = sizeof_fmt(
                        peak_allocation_across_workers
                    )
                    stats_d["input_to_peak_ratio"] = input_to_peak_ratio
                    stats_d["output_to_peak_ratio"] = output_to_peak_ratio
                    stats_d["g_creation_time"] = g_creation_time
                    stats_ls.append(stats_d)
                except Exception as e:
                    print(e)
                restart_client(client)
                sleep(10)

            print("-" * 40 + f"renumber completed" + "-" * 40)
            stats_df = pd.DataFrame(
            stats_ls,
                columns=[
                    "scale",
                    "num_input_edges",
                    "directed",
                    "renumber",
                    "input_memory_per_worker",
                    "peak_allocation_across_workers",
                    "input_to_peak_ratio",
                    "output_to_peak_ratio",
                    "g_creation_time",
                ],
            )
            stats_df.to_csv("cugraph_graph_creation_stats.csv")
        print("-" * 40 + f"scale = {scale} completed" + "-" * 40)
    # Cleanup Dask Cluster
    stop_dask_client(client, cluster)
