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

import logging
import warnings
import argparse
import traceback

from cugraph.testing.mg_utils import (
    generate_edgelist_rmat,
    # get_allocation_counts_dask_persist,
    get_allocation_counts_dask_lazy,
    sizeof_fmt,
    get_peak_output_ratio_across_workers,
    restart_client,
    start_dask_client,
    stop_dask_client,
    enable_spilling,
)

from cugraph.structure.symmetrize import symmetrize
from cugraph.experimental.gnn import BulkSampler

import cugraph

import json
import re
import os
import gc
from time import sleep, perf_counter
from math import ceil

import pandas as pd
import numpy as np
import cupy
import cudf

import dask_cudf
import dask.dataframe as ddf
from dask.distributed import default_client
from cugraph.dask import get_n_workers

from typing import Optional, Union, Dict


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

    if 'etp' in dask_dataframe.columns:
        assert dask_dataframe['etp'].dtype == 'int32'

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_dataframe,
        source="src", 
        destination="dst",
        edge_type='etp' if 'etp' in dask_dataframe.columns else None,
        renumber=False
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

def renumber_ddf(dask_df, persist=False):
    vertices = dask_cudf.concat([dask_df['src'], dask_df['dst']]).unique().reset_index(drop=True)
    if persist:
        vertices = vertices.persist()
    
    vertices.name = 'v'
    vertices = vertices.reset_index().set_index('v').rename(columns={'index': 'm'})
    if persist:
        vertices = vertices.persist()

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


def _replicate_df(df: cudf.DataFrame, replication_factor: int, col_item_counts:Dict[str, int], partition_info: Optional[Union[dict, str]] = None):
    # Required by dask; need to skip dummy partitions.
    if partition_info is None:
        return cudf.DataFrame({
            col: cudf.Series(dtype=df[col].dtype) for col in col_item_counts.keys()
        })
    
    original_df = df.copy()

    if replication_factor > 1:
        for r in range(1, replication_factor):
            df_replicated = original_df
            for col, offset in col_item_counts.items():
                df_replicated[col] += offset * r
        
            df = cudf.concat([df, df_replicated], ignore_index=True)
    
    return df


@get_allocation_counts_dask_lazy(return_allocations=True, logging=True)
def sample_graph(G, label_df, output_path,seed=42, batch_size=500, seeds_per_call=200000, batches_per_partition=100, fanout=[5, 5, 5], persist=False):
    cupy.random.seed(seed)

    sampler = BulkSampler(
        batch_size=batch_size,
        output_path=output_path,
        graph=G,
        fanout_vals=fanout,
        with_replacement=False,
        random_state=seed,
        seeds_per_call=seeds_per_call,
        batches_per_partition=batches_per_partition,
        log_level = logging.INFO
    )

    n_workers = len(default_client().scheduler_info()['workers'])

    meta = cudf.DataFrame({
        'node': cudf.Series(dtype='int64'),
        'batch': cudf.Series(dtype='int32')
    })

    batch_df = label_df.map_partitions(_make_batch_ids, batch_size, n_workers, meta=meta)
    #batch_df = batch_df.sort_values(by='node')
    
    # should always persist the batch dataframe or performance may be suboptimal
    batch_df = batch_df.persist()

    del label_df
    print('created batches')
    

    start_time = perf_counter()
    sampler.add_batches(batch_df, start_col_name='node', batch_col_name='batch')
    sampler.flush()
    end_time = perf_counter()
    print('flushed all batches')
    return (end_time - start_time)


def assign_offsets_pyg(node_counts: Dict[str, int], replication_factor:int=1):
    # cuGraph-PyG assigns offsets based on lexicographic order
    node_offsets = {}
    node_offsets_replicated = {}
    count = 0
    count_replicated = 0
    for node_type in sorted(node_counts.keys()):
        node_offsets[node_type] = count
        node_offsets_replicated[node_type] = count_replicated

        count += node_counts[node_type]
        count_replicated += node_counts[node_type] * replication_factor
    
    return node_offsets, node_offsets_replicated, count_replicated

def generate_rmat_dataset(dataset, seed=62, labeled_percentage=0.01, num_labels=256, reverse_edges=False, persist=False, add_edge_types=False):
    """
    Generates an rmat dataset.  Currently does not support heterogeneous datasets.

    Parameters
    ----------
    dataset: The specifier of the rmat dataset (i.e. rmat_20_16)
    seed: The seed to use for random number generation
    num_labels: The number of classes for the labeled nodes
    reverse_edges: Whether to reverse the edges in the edgelist (should be True for DGL, False, for PyG)
    """

    dataset = dataset.split('_')
    scale = int(dataset[1])
    edgefactor = int(dataset[2])

    dask_edgelist_df = generate_edgelist_rmat(
        scale=scale, edgefactor=edgefactor, seed=seed, unweighted=True, mg=True,
    )
    dask_edgelist_df = dask_edgelist_df.astype("int64")
    dask_edgelist_df = dask_edgelist_df.reset_index(drop=True)


    dask_edgelist_df = renumber_ddf(dask_edgelist_df).persist()
    if persist:
        dask_edgelist_df = dask_edgelist_df.persist()

    dask_edgelist_df = symmetrize_ddf(dask_edgelist_df).persist()
    if persist:
        dask_edgelist_df = dask_edgelist_df.persist()

    if add_edge_types:
        dask_edgelist_df['etp'] = cupy.int32(0) # doesn't matter what the value is, really
    
    # generator = np.random.default_rng(seed=seed)
    num_labeled_nodes = int(2**(scale+1) * labeled_percentage)
    label_df = pd.DataFrame({
        'node': np.arange(num_labeled_nodes),
        # 'label': generator.integers(0, num_labels - 1, num_labeled_nodes).astype('float32')
    })
    
    n_workers = len(default_client().scheduler_info()['workers'])
    dask_label_df = ddf.from_pandas(label_df, npartitions=n_workers*2)
    del label_df
    gc.collect()

    dask_label_df = dask_cudf.from_dask_dataframe(dask_label_df)

    node_offsets = {'paper': 0}
    edge_offsets = {('paper','cites','paper'):0}
    total_num_nodes = int(dask_cudf.concat([dask_edgelist_df.src, dask_edgelist_df.dst]).nunique().compute())

    if reverse_edges:
        dask_edgelist_df = dask_edgelist_df.rename(columns={'src':'dst', 'dst':'src'})

    return dask_edgelist_df, dask_label_df, node_offsets, edge_offsets, total_num_nodes


def load_disk_dataset(dataset, dataset_dir='.', reverse_edges=True, replication_factor=1, persist=False, add_edge_types=False):
    from pathlib import Path
    path = Path(dataset_dir) / dataset
    parquet_path = path / 'parquet'

    n_workers = get_n_workers()

    with open(os.path.join(path, 'meta.json')) as meta_file:
        meta = json.load(meta_file)
    
    node_offsets, node_offsets_replicated, total_num_nodes = \
        assign_offsets_pyg(meta['num_nodes'], replication_factor=replication_factor)

    edge_index_dict = {}
    for edge_type in meta['num_edges'].keys():
        print(f'Loading edge index for edge type {edge_type}')

        can_edge_type = tuple(edge_type.split('__'))
        edge_index_dict[can_edge_type] = dask_cudf.read_parquet(
            Path(parquet_path) / edge_type / 'edge_index.parquet'
        ).repartition(n_workers*2)

        edge_index_dict[can_edge_type]['src'] += node_offsets_replicated[can_edge_type[0]]
        edge_index_dict[can_edge_type]['dst'] += node_offsets_replicated[can_edge_type[-1]]

        edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type]
        if persist:
            edge_index_dict = edge_index_dict.persist()

        if replication_factor > 1:
            edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type].map_partitions(
                _replicate_df,
                replication_factor,
                {
                    'src': meta['num_nodes'][can_edge_type[0]],
                    'dst': meta['num_nodes'][can_edge_type[2]],
                },
                meta=cudf.DataFrame({'src':cudf.Series(dtype='int64'), 'dst':cudf.Series(dtype='int64')})
            )
            
            if persist:
                edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type].persist()
        
        gc.collect()

        if reverse_edges:
            edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type].rename(columns={'src':'dst','dst':'src'})
            
        if persist:
            edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type].persist()
    
    # Assign numeric edge type ids based on lexicographic order
    edge_offsets = {}
    edge_count = 0
    for num_edge_type, can_edge_type in enumerate(sorted(edge_index_dict.keys())):
        if add_edge_types:
            edge_index_dict[can_edge_type]['etp'] = cupy.int32(num_edge_type)
        edge_offsets[can_edge_type] = edge_count
        edge_count += len(edge_index_dict[can_edge_type])
    
    all_edges_df = dask_cudf.concat(
        list(edge_index_dict.values())
    )
    
    if persist:
        all_edges_df = all_edges_df.persist()

    del edge_index_dict
    gc.collect()

    node_labels = {}
    for node_type, offset in node_offsets_replicated.items():
        print(f'Loading node labels for node type {node_type} (offset={offset})')
        node_label_path = os.path.join(os.path.join(parquet_path, node_type), 'node_label.parquet')
        if os.path.exists(node_label_path):
            node_labels[node_type] = dask_cudf.read_parquet(node_label_path).repartition(n_workers).drop('label',axis=1).persist()
            node_labels[node_type]['node'] += offset
            node_labels[node_type] = node_labels[node_type].persist()

            if replication_factor > 1:
                node_labels[node_type] = node_labels[node_type].map_partitions(
                    _replicate_df,
                    replication_factor,
                    {
                        'node': meta['num_nodes'][node_type]
                    },
                    meta=cudf.DataFrame({'node':cudf.Series(dtype='int64')})
                )
                
                if persist:
                    node_labels[node_type] = node_labels[node_type].persist()

            gc.collect()
    
    node_labels_df = dask_cudf.concat(
        list(node_labels.values())
    )
    
    if persist:
        node_labels_df = node_labels_df.persist()

    del node_labels
    gc.collect()

    return all_edges_df, node_labels_df, node_offsets_replicated, edge_offsets, total_num_nodes
    

def benchmark_cugraph_bulk_sampling(
                                    dataset,
                                    output_path,
                                    seed,
                                    batch_size,
                                    seeds_per_call,
                                    fanout,
                                    reverse_edges=True,
                                    dataset_dir='.',
                                    replication_factor=1,
                                    num_labels=256,
                                    labeled_percentage=0.001,
                                    persist=False,
                                    add_edge_types=False):
    """
    Entry point for the benchmark.

    Parameters
    ----------
    dataset: str
        The dataset to sample.  Can be rmat_{scale}_{edgefactor}, or the name of an ogb dataset.
    output_path: str
        The output path, where samples and metadata will be stored.
    seed: int
        The random seed.
    batch_size: int
        The batch size (number of input seeds in a single sampling batch).
    seeds_per_call: int
        The number of input seeds in a single sampling call.
    fanout: list[int]
        The fanout.
    reverse_edges: bool
        Whether to reverse edges when constructing the graph.
    dataset_dir: str
        The directory where datasets are stored (only for ogb datasets)
    replication_factor: int
        The number of times to replicate the dataset.
    num_labels: int
        The number of random labels to generate (only for rmat datasets)
    labeled_percentage: float
        The percentage of the data that is labeled (only for rmat datasets)
        Defaults to 0.001 to match papers100M
    persist: bool
        Whether to aggressively persist data in dask in attempt to speed up ETL.
        Defaults to False.
    add_edge_types: bool
        Whether to add edge types to the edgelist.
        Defaults to False.
    """
    print(dataset)
    if dataset[0:4] == 'rmat':
        dask_edgelist_df, dask_label_df, node_offsets, edge_offsets, total_num_nodes = \
            generate_rmat_dataset(
                dataset,
                reverse_edges=reverse_edges,
                seed=seed,
                labeled_percentage=labeled_percentage,
                num_labels=num_labels,
                persist=persist,
                add_edge_types=add_edge_types
            )

    else:
        dask_edgelist_df, dask_label_df, node_offsets, edge_offsets, total_num_nodes = \
            load_disk_dataset(
                dataset,
                dataset_dir=dataset_dir,
                reverse_edges=reverse_edges,
                replication_factor=replication_factor,
                persist=persist,
                add_edge_types=add_edge_types
            )

    num_input_edges = len(dask_edgelist_df)
    print(
        f"Number of input edges = {num_input_edges:,}"
    )

    G = construct_graph(
        dask_edgelist_df
    )
    del dask_edgelist_df
    print('constructed graph')

    input_memory = G.edgelist.edgelist_df.memory_usage().sum().compute()
    print(f'input memory: {input_memory}')

    output_subdir = os.path.join(output_path, f'{dataset}[{replication_factor}]_b{batch_size}_f{fanout}')
    os.makedirs(output_subdir)

    output_sample_path = os.path.join(output_subdir, 'samples')
    os.makedirs(output_sample_path)

    batches_per_partition = 200_000 // batch_size
    execution_time, allocation_counts = sample_graph(
        G,
        dask_label_df,
        output_sample_path,
        seed=seed,
        batch_size=batch_size,
        seeds_per_call=seeds_per_call,
        batches_per_partition=batches_per_partition,
        fanout=fanout,
        persist=persist,
    )

    output_meta = {
        'dataset': dataset,
        'dataset_dir': dataset_dir,
        'seed': seed,
        'node_offsets': node_offsets,
        'edge_offsets': {'__'.join(k): v for k, v in edge_offsets.items()},
        'total_num_nodes': total_num_nodes,
        'total_num_edges': num_input_edges,
        'batch_size': batch_size,
        'seeds_per_call': seeds_per_call,
        'batches_per_partition': batches_per_partition,
        'fanout': fanout,
        'replication_factor': replication_factor,
        'num_sampling_gpus': len(G._plc_graph),
        'execution_time': execution_time,
    }

    with open(os.path.join(output_subdir, 'output_meta.json'), 'w') as f:
        json.dump(
            output_meta,
            f,
            indent='\t'
        )

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


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--output_root',
        type=str,
        help='The output root directory.  File/folder names are auto-generated.',
        required=True,
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        help='The dataset root directory containing ogb datasets.',
        required=True,
    )

    parser.add_argument(
        '--datasets',
        type=str,
        help=(
            'Comma separated list of datasets; can specify ogb or rmat (i.e. ogb_papers100M[2],rmat_22_16).'
            ' For ogb datasets, can provide replication factor using brackets.'
        ),
        required=True,
    )

    parser.add_argument(
        '--fanouts',
        type=str,
        help='Comma separated list of fanouts (i.e. 10_25,5_5_5)',
        required=False,
        default='10_25',
    )

    parser.add_argument(
        '--batch_sizes',
        type=str,
        help='Comma separated list of batch sizes (i.e. 500,1000)',
        required=False,
        default='512,1024'
    )

    parser.add_argument(
        '--seeds_per_call_opts',
        type=str,
        help='Comma separated list of seeds per call (i.e. 1000000,2000000)',
        required=False,
        default='524288',
    )
    
    parser.add_argument(
        '--reverse_edges',
        action='store_true',
        help='Whether to reverse the edges for DGL (defaults to False).  Should be True for DGL, False for PyG.',
        required=False,
        default=False,
    )

    parser.add_argument(
        '--dask_worker_devices',
        type=str,
        help='Comma separated list of dask worker devices',
        required=False,
        default="0"
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        help='Random seed',
        required=False,
        default=62
    )

    parser.add_argument(
        '--persist',
        action='store_true',
        help='Will add additional persist() calls to speed up ETL.  Does not affect sampling runtime.',
        required=False,
        default=False,
    )

    parser.add_argument(
        '--add_edge_types',
        action='store_true',
        help='Adds edge types to the edgelist.  Required for PyG if not providing edge ids.',
        required=False,
        default=False,
    )

    return parser.parse_args()


# call __main__ function
if __name__ == "__main__":
    logging.basicConfig()

    args = get_args()
    fanouts = [[int(f) for f in fanout.split('_')] for fanout in args.fanouts.split(',')]
    datasets = args.datasets.split(',')
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]
    seeds_per_call_opts = [int(s) for s in args.seeds_per_call_opts.split(',')]
    dask_worker_devices = [int(d) for d in args.dask_worker_devices.split(',')]

    client, cluster = start_dask_client(dask_worker_devices=dask_worker_devices, jit_unspill=False, rmm_pool_size=28e9, rmm_async=True)
    enable_spilling()
    stats_ls = []
    client.run(enable_spilling)
    for dataset in datasets:
        if re.match(r'([A-z]|[0-9])+\[[0-9]+\]', dataset):
            replication_factor = int(dataset[-2])
            dataset = dataset[:-3]
        else:
            replication_factor = 1

        for fanout in fanouts:
            for batch_size in batch_sizes:
                for seeds_per_call in seeds_per_call_opts:
                    print(f'dataset: {dataset}')
                    print(f'batch size: {batch_size}')
                    print(f'fanout: {fanout}')
                    print(f'seeds_per_call: {seeds_per_call}')

                    try:
                        stats_d = {}
                        (
                            num_input_edges,
                            input_to_peak_ratio,
                            output_to_peak_ratio,
                            input_memory_per_worker,
                            peak_allocation_across_workers,
                        ) = benchmark_cugraph_bulk_sampling(
                            dataset=dataset,
                            output_path=args.output_root,
                            seed=args.random_seed,
                            batch_size=batch_size,
                            seeds_per_call=seeds_per_call,
                            fanout=fanout,
                            dataset_dir=args.dataset_root,
                            reverse_edges=args.reverse_edges,
                            replication_factor=replication_factor,
                            persist=args.persist,
                            add_edge_types=args.add_edge_types,
                        )
                        stats_d["dataset"] = dataset
                        stats_d["num_input_edges"] = num_input_edges
                        stats_d["batch_size"] = batch_size
                        stats_d["fanout"] = fanout
                        stats_d["seeds_per_call"] = seeds_per_call
                        stats_d["input_memory_per_worker"] = sizeof_fmt(input_memory_per_worker)
                        stats_d["peak_allocation_across_workers"] = sizeof_fmt(
                            peak_allocation_across_workers
                        )
                        stats_d["input_to_peak_ratio"] = input_to_peak_ratio
                        stats_d["output_to_peak_ratio"] = output_to_peak_ratio
                        stats_ls.append(stats_d)
                    except Exception as e:
                        warnings.warn('An Exception Occurred!')
                        print(e)
                        traceback.print_exc()
                    restart_client(client)
                    sleep(10)

        stats_df = pd.DataFrame(
            stats_ls,
            columns=[
                "dataset",
                "num_input_edges",
                "directed",
                "renumber",
                "input_memory_per_worker",
                "peak_allocation_across_workers",
                "input_to_peak_ratio",
                "output_to_peak_ratio",
            ],
        )
        stats_df.to_csv("cugraph_sampling_stats.csv")
        print("-" * 40 + f"dataset = {dataset} completed" + "-" * 40)

    # Cleanup Dask Cluster
    stop_dask_client(client, cluster)
