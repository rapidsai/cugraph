# Copyright (c) 2023, NVIDIA CORPORATION.
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
    sizeof_fmt,
    get_peak_output_ratio_across_workers,
    restart_client,
)

from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
    enable_spilling,
)

from cugraph.structure.symmetrize import symmetrize
from cugraph.experimental.gnn import BulkSampler

from cugraph.dask import uniform_neighbor_sample

import cugraph

from time import sleep
from math import ceil

import pandas as pd
import cudf
import dask_cudf
import cupy

from typing import Optional, Union


def construct_graph(dask_dataframe):
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
    assert dask_dataframe['src'].dtype == 'int64'
    assert dask_dataframe['dst'].dtype == 'int64'
    assert dask_dataframe['etp'].dtype == 'int32'

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_dataframe, source="src", destination="dst", edge_type='etp', renumber=False
    )
    return G


def symmetrize_ddf(dask_dataframe):
    source_col, dest_col = symmetrize(
        dask_dataframe,
        'src',
        'dst',
        multi=True,
        symmetrize=True,
    )

    new_ddf = source_col.to_frame()
    new_ddf['dst'] = dest_col

    return new_ddf

def renumber_ddf(dask_df):
    vertices = dask_cudf.concat([dask_df['src'], dask_df['dst']]).unique().reset_index(drop=True).persist()
    vertices.name = 'v'
    vertices = vertices.reset_index().set_index('v').rename(columns={'index': 'm'}).persist()

    src = dask_df.merge(vertices, left_on='src', right_on='v', how='left').m.rename('src')
    dst = dask_df.merge(vertices, left_on='dst', right_on='v', how='left').m.rename('dst')
    df = src.to_frame()
    df['dst'] = dst

    return df.reset_index(drop=True)

def _make_batch_ids(bdf: cudf.DataFrame, batch_size: int, num_workers: int, partition_info: Optional[Union[dict, str]] = None):
    # Required by dask; need to skip dummy partitions.
    if partition_info is None:
        return cudf.DataFrame({
            'batch': cudf.Series(dtype='int32'),
            'start': cudf.Series(dtype='int64')
        })
    
    partition = partition_info['number']
    if partition is None:
        raise ValueError('division is absent')

    num_batches = int(ceil(len(bdf) / batch_size))
    
    batch_ids = cupy.repeat(
        cupy.arange(num_batches * partition, num_batches * (partition + 1), dtype='int32'),
        batch_size
    )[:len(bdf)]

    bdf = bdf.reset_index(drop=True)
    bdf['batch'] = cudf.Series(batch_ids)

    return bdf


@get_allocation_counts_dask_persist(return_allocations=True, logging=True)
def sample_graph(G, seed=42, batch_size=500, fanout=[5, 5, 5]):
    cupy.random.seed(seed)

    sampler = BulkSampler(
        batch_size=batch_size,
        output_path='/tmp/samples',
        graph=G,
        fanout_vals=fanout,
        with_replacement=False,
        random_state=seed,
    )

    from dask.distributed import wait, default_client
    n_workers = len(default_client().scheduler_info()['workers'])

    meta = cudf.DataFrame({
        'start': cudf.Series(dtype='int64'),
        'batch': cudf.Series(dtype='int32')
    })

    
    batch_df = dask_cudf.concat([G.edgelist.edgelist_df['src'], G.edgelist.edgelist_df['dst']]).unique().rename('start').to_frame().reset_index(drop=True).persist()
    batch_df = batch_df.map_partitions(_make_batch_ids, batch_size, n_workers, meta=meta).persist()
    print('created batches')
    

    sampler.add_batches(batch_df, start_col_name='start', batch_col_name='batch')
    sampler.flush()
    print('flushed all batches')
    
    """    
    results_ddf = uniform_neighbor_sample(
        G,
        batch_df.start,
        fanout_vals=[10,25],
        batch_id_list=batch_df.batch,
        with_replacement=False,
        with_edge_properties=True,
        random_state=seed
    )
    print(results_ddf.compute())
    """

def benchmark_cugraph_bulk_sampling(scale, edgefactor, seed, batch_size, fanout):
    """
    Entry point for the benchmark.
    """
    dask_df = generate_edgelist_rmat(
        scale=scale, edgefactor=edgefactor, seed=seed, unweighted=True, mg=True,
    )
    dask_df = dask_df.astype("int64")
    dask_df = dask_df.reset_index(drop=True)


    dask_df = renumber_ddf(dask_df).persist()
    dask_df = symmetrize_ddf(dask_df).persist()
    dask_df['etp'] = cupy.int32(0) # doesn't matter what the value is, really

    num_input_edges = len(dask_df)
    print(
        f"Number of input edges = {num_input_edges:,}"
    )

    G = construct_graph(
        dask_df
    )
    print('constructed graph')

    input_memory = G.edgelist.edgelist_df.memory_usage().sum().compute()
    print(f'input memory: {input_memory}')

    _, allocation_counts = sample_graph(G, seed, batch_size, fanout)
    print('allocation counts b:')
    print(allocation_counts.values())

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

# call __main__ function
if __name__ == "__main__":
    client, cluster = start_dask_client(dask_worker_devices=[1], jit_unspill=False)
    enable_spilling()
    stats_ls = []
    client.run(enable_spilling)
    #for scale in [22, 23, 24]:
    for scale in [16, 22, 24]:
        for fanout in [[10,25]]:
            for batch_size in [500, 1000]:
                print(f'scale: {scale}')
                print(f'batch size: {batch_size}')
                print(f'fanout: {fanout}')

                try:
                    stats_d = {}
                    (
                        num_input_edges,
                        input_to_peak_ratio,
                        output_to_peak_ratio,
                        input_memory_per_worker,
                        peak_allocation_across_workers,
                    ) = benchmark_cugraph_bulk_sampling(
                        scale=scale,
                        edgefactor=16,
                        seed=123,
                        batch_size=batch_size,
                        fanout=fanout,
                    )
                    stats_d["scale"] = scale
                    stats_d["num_input_edges"] = num_input_edges
                    stats_d["batch_size"] = batch_size
                    stats_d["fanout"] = fanout
                    stats_d["input_memory_per_worker"] = sizeof_fmt(input_memory_per_worker)
                    stats_d["peak_allocation_across_workers"] = sizeof_fmt(
                        peak_allocation_across_workers
                    )
                    stats_d["input_to_peak_ratio"] = input_to_peak_ratio
                    stats_d["output_to_peak_ratio"] = output_to_peak_ratio
                    stats_ls.append(stats_d)
                except Exception as e:
                    print(e)
                restart_client(client)
                sleep(10)

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
            ],
        )
        stats_df.to_csv("cugraph_graph_creation_stats.csv")
        print("-" * 40 + f"scale = {scale} completed" + "-" * 40)

    # Cleanup Dask Cluster
    stop_dask_client(client, cluster)
