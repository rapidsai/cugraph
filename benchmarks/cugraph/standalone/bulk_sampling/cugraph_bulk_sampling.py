# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
    start_dask_client,
    stop_dask_client,
    enable_spilling,
)

from cugraph.structure.symmetrize import symmetrize
from cugraph.gnn import BulkSampler

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
    assert dask_dataframe["src"].dtype == "int64"
    assert dask_dataframe["dst"].dtype == "int64"

    if "etp" in dask_dataframe.columns:
        assert dask_dataframe["etp"].dtype == "int32"

    G = cugraph.MultiGraph(directed=True)
    G.from_dask_cudf_edgelist(
        dask_dataframe,
        source="src",
        destination="dst",
        edge_type="etp" if "etp" in dask_dataframe.columns else None,
        renumber=False,
    )
    return G


def symmetrize_ddf(dask_dataframe):
    source_col, dest_col = symmetrize(
        dask_dataframe,
        "src",
        "dst",
        multi=True,
        symmetrize=True,
    )

    new_ddf = source_col.to_frame()
    new_ddf["dst"] = dest_col

    return new_ddf


def renumber_ddf(dask_df):
    vertices = (
        dask_cudf.concat([dask_df["src"], dask_df["dst"]])
        .unique()
        .reset_index(drop=True)
    )

    vertices.name = "v"
    vertices = vertices.reset_index().set_index("v").rename(columns={"index": "m"})

    src = dask_df.merge(vertices, left_on="src", right_on="v", how="left").m.rename(
        "src"
    )
    dst = dask_df.merge(vertices, left_on="dst", right_on="v", how="left").m.rename(
        "dst"
    )
    df = src.to_frame()
    df["dst"] = dst

    return df.reset_index(drop=True)


def _make_batch_ids(
    bdf: cudf.DataFrame,
    batch_size: int,
    num_workers: int,
    partition_info: Optional[Union[dict, str]] = None,
):
    # Required by dask; need to skip dummy partitions.
    if partition_info is None:
        return cudf.DataFrame(
            {"batch": cudf.Series(dtype="int32"), "start": cudf.Series(dtype="int64")}
        )

    partition = partition_info["number"]
    if partition is None:
        raise ValueError("division is absent")

    num_batches = int(ceil(len(bdf) / batch_size))

    batch_ids = cupy.repeat(
        cupy.arange(
            num_batches * partition, num_batches * (partition + 1), dtype="int32"
        ),
        batch_size,
    )[: len(bdf)]

    bdf = bdf.reset_index(drop=True)
    bdf["batch"] = cudf.Series(batch_ids)

    return bdf


def _replicate_df(
    df: cudf.DataFrame,
    replication_factor: int,
    col_item_counts: Dict[str, int],
    partition_info: Optional[Union[dict, str]] = None,
):
    # Required by dask; need to skip dummy partitions.
    if partition_info is None:
        return cudf.DataFrame(
            {col: cudf.Series(dtype=df[col].dtype) for col in col_item_counts.keys()}
        )

    original_df = df.copy()

    if replication_factor > 1:
        for r in range(1, replication_factor):
            df_replicated = original_df.copy()
            for col, offset in col_item_counts.items():
                df_replicated[col] += offset * r

            df = cudf.concat([df, df_replicated], ignore_index=True)

    return df


@get_allocation_counts_dask_lazy(return_allocations=True, logging=True)
def sample_graph(
    G,
    label_df,
    output_path,
    seed=42,
    batch_size=500,
    seeds_per_call=400000,
    batches_per_partition=100,
    fanout=[5, 5, 5],
    num_epochs=1,
    train_perc=0.8,
    val_perc=0.5,
    sampling_kwargs={},
):
    logger = logging.getLogger("__main__")
    logger.info("Starting sampling phase...")

    logger.info("Calculating random splits...")
    cupy.random.seed(seed)
    train_df, test_df = label_df.random_split(
        [train_perc, 1 - train_perc], random_state=seed, shuffle=True
    )
    val_df, test_df = label_df.random_split(
        [val_perc, 1 - val_perc], random_state=seed, shuffle=True
    )
    logger.info("Calculated random splits")

    total_time = 0.0
    for epoch in range(num_epochs):
        steps = [("train", train_df)]
        if epoch == num_epochs - 1:
            steps.append(("val", val_df))
            steps.append(("test", test_df))

        for step, batch_df in steps:
            logger.info("Shuffling batch dataframe...")
            batch_df = batch_df.sample(frac=1.0, random_state=seed).persist()
            logger.info("Shuffled and persisted batch dataframe...")

            if step == "train":
                output_sample_path = os.path.join(
                    output_path, f"epoch={epoch}", f"{step}", "samples"
                )
            else:
                output_sample_path = os.path.join(output_path, step, "samples")

            client = default_client()

            def func():
                os.makedirs(output_sample_path, exist_ok=True)

            client.run(func)

            logger.info("Creating bulk sampler...")
            sampler = BulkSampler(
                batch_size=batch_size,
                output_path=output_sample_path,
                graph=G,
                fanout_vals=fanout,
                with_replacement=False,
                random_state=seed,
                seeds_per_call=seeds_per_call,
                batches_per_partition=batches_per_partition,
                log_level=logging.INFO,
                **sampling_kwargs,
            )
            logger.info("Bulk sampler created and ready for input")

            n_workers = len(default_client().scheduler_info()["workers"])

            meta = cudf.DataFrame(
                {
                    "node": cudf.Series(dtype="int64"),
                    "batch": cudf.Series(dtype="int32"),
                }
            )

            batch_df = batch_df.map_partitions(
                _make_batch_ids, batch_size, n_workers, meta=meta
            )

            # should always persist the batch dataframe or performance may be suboptimal
            batch_df = batch_df.persist()

            logger.info("created and persisted batches")

            start_time = perf_counter()
            sampler.add_batches(batch_df, start_col_name="node", batch_col_name="batch")
            sampler.flush()
            end_time = perf_counter()
            logger.info("flushed all batches")
            total_time += end_time - start_time

    return total_time


def assign_offsets_pyg(node_counts: Dict[str, int], replication_factor: int = 1):
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


def generate_rmat_dataset(
    dataset,
    seed=62,
    labeled_percentage=0.01,
    num_labels=256,
    reverse_edges=False,
    add_edge_types=False,
):
    """
    Generates an rmat dataset.  Currently does not support heterogeneous datasets.

    Parameters
    ----------
    dataset: The specifier of the rmat dataset (i.e. rmat_20_16)
    seed: The seed to use for random number generation
    num_labels: The number of classes for the labeled nodes
    reverse_edges: Whether to reverse the edges in the edgelist (should be True for DGL, False, for PyG)
    """

    dataset = dataset.split("_")
    scale = int(dataset[1])
    edgefactor = int(dataset[2])

    dask_edgelist_df = generate_edgelist_rmat(
        scale=scale,
        edgefactor=edgefactor,
        seed=seed,
        unweighted=True,
        mg=True,
    )
    dask_edgelist_df = dask_edgelist_df.astype("int64")
    dask_edgelist_df = dask_edgelist_df.reset_index(drop=True)

    dask_edgelist_df = renumber_ddf(dask_edgelist_df).persist()

    dask_edgelist_df = symmetrize_ddf(dask_edgelist_df).persist()

    if add_edge_types:
        dask_edgelist_df["etp"] = cupy.int32(
            0
        )  # doesn't matter what the value is, really

    # generator = np.random.default_rng(seed=seed)
    num_labeled_nodes = int(2 ** (scale + 1) * labeled_percentage)
    label_df = pd.DataFrame(
        {
            "node": np.arange(num_labeled_nodes),
            # 'label': generator.integers(0, num_labels - 1, num_labeled_nodes).astype('float32')
        }
    )

    n_workers = len(default_client().scheduler_info()["workers"])
    dask_label_df = ddf.from_pandas(label_df, npartitions=n_workers * 2)
    del label_df
    gc.collect()

    dask_label_df = dask_cudf.from_dask_dataframe(dask_label_df)

    node_offsets = {"paper": 0}
    edge_offsets = {("paper", "cites", "paper"): 0}
    total_num_nodes = int(
        dask_cudf.concat([dask_edgelist_df.src, dask_edgelist_df.dst])
        .nunique()
        .compute()
    )

    if reverse_edges:
        dask_edgelist_df = dask_edgelist_df.rename(columns={"src": "dst", "dst": "src"})

    return dask_edgelist_df, dask_label_df, node_offsets, edge_offsets, total_num_nodes


def load_disk_dataset(
    dataset,
    dataset_dir=".",
    reverse_edges=True,
    replication_factor=1,
    add_edge_types=False,
):
    from pathlib import Path

    path = Path(dataset_dir) / dataset
    parquet_path = path / "parquet"

    logger = logging.getLogger("__main__")

    logger.info("getting n workers...")
    n_workers = get_n_workers()
    logger.info(f"there are {n_workers} workers")

    with open(os.path.join(path, "meta.json")) as meta_file:
        meta = json.load(meta_file)

    logger.info("assigning offsets...")
    node_offsets, node_offsets_replicated, total_num_nodes = assign_offsets_pyg(
        meta["num_nodes"], replication_factor=replication_factor
    )
    logger.info("offsets assigned")

    edge_index_dict = {}
    for edge_type in meta["num_edges"].keys():
        logger.info(f"Loading edge index for edge type {edge_type}")

        can_edge_type = tuple(edge_type.split("__"))
        edge_index_dict[can_edge_type] = dask_cudf.read_parquet(
            Path(parquet_path) / edge_type / "edge_index.parquet"
        ).repartition(npartitions=n_workers * 2)

        edge_index_dict[can_edge_type]["src"] += node_offsets_replicated[
            can_edge_type[0]
        ]
        edge_index_dict[can_edge_type]["dst"] += node_offsets_replicated[
            can_edge_type[-1]
        ]

        edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type]

        if replication_factor > 1:
            logger.info("processing replications")
            edge_index_dict[can_edge_type] = edge_index_dict[
                can_edge_type
            ].map_partitions(
                _replicate_df,
                replication_factor,
                {
                    "src": meta["num_nodes"][can_edge_type[0]],
                    "dst": meta["num_nodes"][can_edge_type[2]],
                },
                meta=cudf.DataFrame(
                    {
                        "src": cudf.Series(dtype="int64"),
                        "dst": cudf.Series(dtype="int64"),
                    }
                ),
            )
            logger.info("replications processed")

        gc.collect()

        if reverse_edges:
            edge_index_dict[can_edge_type] = edge_index_dict[can_edge_type].rename(
                columns={"src": "dst", "dst": "src"}
            )
        logger.info("edge index loaded")

    # Assign numeric edge type ids based on lexicographic order
    edge_offsets = {}
    edge_count = 0
    # for num_edge_type, can_edge_type in enumerate(sorted(edge_index_dict.keys())):
    #    if add_edge_types:
    #        edge_index_dict[can_edge_type]["etp"] = cupy.int32(num_edge_type)
    #    edge_offsets[can_edge_type] = edge_count
    #    edge_count += len(edge_index_dict[can_edge_type])

    if len(edge_index_dict) != 1:
        raise ValueError("should only be 1 edge index")

    logger.info("setting edge type")

    all_edges_df = list(edge_index_dict.values())[0]
    if add_edge_types:
        all_edges_df["etp"] = cupy.int32(0)

    # all_edges_df = dask_cudf.concat(list(edge_index_dict.values()))

    del edge_index_dict
    gc.collect()

    node_labels = {}
    for node_type, offset in node_offsets_replicated.items():
        logger.info(f"Loading node labels for node type {node_type} (offset={offset})")
        node_label_path = os.path.join(
            os.path.join(parquet_path, node_type), "node_label.parquet"
        )
        if os.path.exists(node_label_path):
            node_labels[node_type] = (
                dask_cudf.read_parquet(node_label_path)
                .repartition(npartitions=n_workers)
                .drop("label", axis=1)
                .persist()
            )
            logger.info(f"Loaded and persisted initial labels")
            node_labels[node_type]["node"] += offset
            node_labels[node_type] = node_labels[node_type].persist()
            logger.info(f"Set and persisted node offsets")

            if replication_factor > 1:
                logger.info(f"Replicating labels...")
                node_labels[node_type] = node_labels[node_type].map_partitions(
                    _replicate_df,
                    replication_factor,
                    {"node": meta["num_nodes"][node_type]},
                    meta=cudf.DataFrame({"node": cudf.Series(dtype="int64")}),
                )
                logger.info(f"Replicated labels (will likely evaluate later)")

            gc.collect()

    node_labels_df = dask_cudf.concat(list(node_labels.values())).reset_index(drop=True)
    logger.info("Dataset successfully loaded")

    del node_labels
    gc.collect()

    return (
        all_edges_df,
        node_labels_df,
        node_offsets_replicated,
        edge_offsets,
        total_num_nodes,
        sum(meta["num_edges"].values()) * replication_factor,
    )


def benchmark_cugraph_bulk_sampling(
    dataset,
    output_path,
    seed,
    batch_size,
    seeds_per_call,
    fanout,
    sampling_target_framework,
    reverse_edges=True,
    dataset_dir=".",
    replication_factor=1,
    num_labels=256,
    labeled_percentage=0.001,
    add_edge_types=False,
    num_epochs=1,
):
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
    add_edge_types: bool
        Whether to add edge types to the edgelist.
        Defaults to False.
    sampling_target_framework: str
        The framework to sample for.
    num_epochs: int
        The number of epochs to sample for.
    """

    logger = logging.getLogger("__main__")
    logger.info(str(dataset))
    if dataset[0:4] == "rmat":
        (
            dask_edgelist_df,
            dask_label_df,
            node_offsets,
            edge_offsets,
            total_num_nodes,
        ) = generate_rmat_dataset(
            dataset,
            reverse_edges=reverse_edges,
            seed=seed,
            labeled_percentage=labeled_percentage,
            num_labels=num_labels,
            add_edge_types=add_edge_types,
        )

    else:
        (
            dask_edgelist_df,
            dask_label_df,
            node_offsets,
            edge_offsets,
            total_num_nodes,
            num_input_edges,
        ) = load_disk_dataset(
            dataset,
            dataset_dir=dataset_dir,
            reverse_edges=reverse_edges,
            replication_factor=replication_factor,
            add_edge_types=add_edge_types,
        )

    logger.info(f"Number of input edges = {num_input_edges:,}")

    G = construct_graph(dask_edgelist_df)
    del dask_edgelist_df
    logger.info("constructed graph")

    input_memory = G.edgelist.edgelist_df.memory_usage().sum().compute()
    logger.info(f"input memory: {input_memory}")

    output_subdir = os.path.join(
        output_path,
        f"{dataset}[{replication_factor}]_b{batch_size}_f{fanout}",
    )

    client = default_client()

    def func():
        os.makedirs(output_subdir, exist_ok=True)

    client.run(func)

    if sampling_target_framework == "cugraph_dgl_csr":
        sampling_kwargs = {
            "deduplicate_sources": True,
            "prior_sources_behavior": "carryover",
            "renumber": True,
            "compression": "CSR",
            "compress_per_hop": True,
            "use_legacy_names": False,
            "include_hop_column": False,
        }
    elif sampling_target_framework == "cugraph_pyg":
        # FIXME: Update these arguments when CSC mode is fixed in cuGraph-PyG (release 24.04)
        sampling_kwargs = {
            "deduplicate_sources": True,
            "prior_sources_behavior": "exclude",
            "renumber": True,
            "compression": "COO",
            "compress_per_hop": False,
            "use_legacy_names": False,
            "include_hop_column": True,
        }
    else:
        raise ValueError("Only cugraph_dgl_csr or cugraph_pyg are valid frameworks")

    batches_per_partition = 256
    execution_time, allocation_counts = sample_graph(
        G=G,
        label_df=dask_label_df,
        output_path=output_subdir,
        num_epochs=num_epochs,
        seed=seed,
        batch_size=batch_size,
        seeds_per_call=seeds_per_call,
        batches_per_partition=batches_per_partition,
        fanout=fanout,
        sampling_kwargs=sampling_kwargs,
    )

    output_meta = {
        "dataset": dataset,
        "dataset_dir": dataset_dir,
        "seed": seed,
        "node_offsets": node_offsets,
        "edge_offsets": {"__".join(k): v for k, v in edge_offsets.items()},
        "total_num_nodes": total_num_nodes,
        "total_num_edges": num_input_edges,
        "batch_size": batch_size,
        "seeds_per_call": seeds_per_call,
        "batches_per_partition": batches_per_partition,
        "fanout": fanout,
        "replication_factor": replication_factor,
        "num_sampling_gpus": len(G._plc_graph),
        "execution_time": execution_time,
    }

    with open(os.path.join(output_subdir, "output_meta.json"), "w") as f:
        json.dump(output_meta, f, indent="\t")

    logger.info("allocation counts b:")
    logger.info(allocation_counts.values())

    (
        input_to_peak_ratio,
        output_to_peak_ratio,
        input_memory_per_worker,
        peak_allocation_across_workers,
    ) = get_memory_statistics(
        allocation_counts=allocation_counts, input_memory=input_memory
    )
    logger.info(f"Number of edges in final graph = {G.number_of_edges():,}")
    logger.info("-" * 80)
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
        "--output_root",
        type=str,
        help="The output root directory.  File/folder names are auto-generated.",
        required=True,
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        help="The dataset root directory containing ogb datasets.",
        required=True,
    )

    parser.add_argument(
        "--datasets",
        type=str,
        help=(
            "Comma separated list of datasets; can specify ogb or rmat (i.e. ogb_papers100M[2],rmat_22_16)."
            " For ogb datasets, can provide replication factor using brackets."
        ),
        required=True,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to run for",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--fanouts",
        type=str,
        help='Comma separated list of fanouts (i.e. "10_25,5_5_5")',
        required=False,
        default="10_10_10",
    )

    parser.add_argument(
        "--batch_sizes",
        type=str,
        help="Comma separated list of batch sizes (i.e. 500,1000)",
        required=False,
        default="512,1024",
    )

    parser.add_argument(
        "--seeds_per_call_opts",
        type=str,
        help="Comma separated list of seeds per call (i.e. 1000000,2000000)",
        required=False,
        default="524288",
    )

    parser.add_argument(
        "--reverse_edges",
        action="store_true",
        help="Whether to reverse the edges for DGL (defaults to False).  Should be True for DGL, False for PyG.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--sampling_target_framework",
        type=str,
        help="The target framework for sampling (i.e. cugraph_dgl_csr, cugraph_pyg_csc, ...)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--dask_worker_devices",
        type=str,
        help="Comma separated list of dask worker devices",
        required=False,
        default="0",
    )

    parser.add_argument(
        "--random_seed", type=int, help="Random seed", required=False, default=62
    )

    return parser.parse_args()


# call __main__ function
if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("__main__")
    logger.setLevel(logging.INFO)

    args = get_args()
    if args.sampling_target_framework not in ["cugraph_dgl_csr", "cugraph_pyg"]:
        raise ValueError(
            "sampling_target_framework must be one of cugraph_dgl_csr or cugraph_pyg",
            "Other frameworks are not supported at this time.",
        )

    fanouts = [
        [int(f) for f in fanout.split("_")] for fanout in args.fanouts.split(",")
    ]
    datasets = args.datasets.split(",")
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    seeds_per_call_opts = [int(s) for s in args.seeds_per_call_opts.split(",")]
    dask_worker_devices = [int(d) for d in args.dask_worker_devices.split(",")]

    import time

    time_dask_start = time.localtime()

    logger.info(f"{time.asctime(time_dask_start)}: starting dask client")
    from dask_cuda.initialize import initialize
    from dask.distributed import Client
    from cugraph.dask.comms import comms as Comms
    import os, time

    client = Client(scheduler_file=os.environ["SCHEDULER_FILE"], timeout=360)
    time.sleep(30)
    cluster = Comms.initialize(p2p=True)
    # client, cluster = start_dask_client()
    time_dask_end = time.localtime()
    logger.info(f"{time.asctime(time_dask_end)}: dask client started")

    logger.info("enabling spilling")
    enable_spilling()
    client.run(enable_spilling)
    logger.info("enabled spilling")

    stats_ls = []

    for dataset in datasets:
        m = re.match(r"(\w+)\[([0-9]+)\]", dataset)
        if m:
            replication_factor = int(m.groups()[1])
            dataset = m.groups()[0]
        else:
            replication_factor = 1

        for fanout in fanouts:
            for batch_size in batch_sizes:
                for seeds_per_call in seeds_per_call_opts:
                    logger.info(f"dataset: {dataset}")
                    logger.info(f"batch size: {batch_size}")
                    logger.info(f"fanout: {fanout}")
                    logger.info(f"seeds_per_call: {seeds_per_call}")
                    logger.info(f"num epochs: {args.num_epochs}")

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
                            num_epochs=args.num_epochs,
                            seed=args.random_seed,
                            batch_size=batch_size,
                            seeds_per_call=seeds_per_call,
                            fanout=fanout,
                            sampling_target_framework=args.sampling_target_framework,
                            dataset_dir=args.dataset_root,
                            reverse_edges=args.reverse_edges,
                            replication_factor=replication_factor,
                        )
                        stats_d["dataset"] = dataset
                        stats_d["num_input_edges"] = num_input_edges
                        stats_d["batch_size"] = batch_size
                        stats_d["fanout"] = fanout
                        stats_d["seeds_per_call"] = seeds_per_call
                        stats_d["input_memory_per_worker"] = sizeof_fmt(
                            input_memory_per_worker
                        )
                        stats_d["peak_allocation_across_workers"] = sizeof_fmt(
                            peak_allocation_across_workers
                        )
                        stats_d["input_to_peak_ratio"] = input_to_peak_ratio
                        stats_d["output_to_peak_ratio"] = output_to_peak_ratio
                        stats_ls.append(stats_d)
                    except Exception as e:
                        warnings.warn("An Exception Occurred!")
                        print(e)
                        traceback.print_exc()
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
