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
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf


import time
import os
import pytest
import numpy as np
import cupy as cp
# Facing issues with rapids-pytest-benchmark plugin
# pytest-benchmark.
import pytest_benchmark

from cugraph.generators import rmat
from cugraph.experimental import datasets
from cugraph_benchmarking import params
from cugraph_dgl import CuGraphStorage
from cugraph.dask.comms import comms as Comms
import dgl
import torch
import rmm
_seed = 42


def create_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """    
    print("Initalize Pool on client")
    rmm.reinitialize(pool_allocator=True)
    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()

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

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    num_nodes = max(edgelist_df['src'].max(),
                edgelist_df['dst'].max())+1

    num_nodes_dict = {'_N':num_nodes}

    gs = CuGraphStorage(num_nodes_dict=num_nodes_dict, single_gpu=True)
    gs.add_edge_data(edgelist_df,   
                    # reverse to make same graph as cugraph
                    node_col_names=['dst', 'src'],
                    canonical_etype=['_N', 'connects', '_N'])

    return gs



def create_mg_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
    ## Reserving GPU 0 for client(trainer/service project)
    n_devices = os.getenv('DASK_NUM_WORKERS', 4)
    n_devices = int(n_devices)

    visible_devices = ','.join([str(i) for i in range(1, n_devices+1)])
    cluster = LocalCUDACluster(protocol='ucx', rmm_pool_size='25GB', CUDA_VISIBLE_DEVICES=visible_devices)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    rmm.reinitialize(pool_allocator=True)
    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        edgelist_df = dask_cudf.from_cudf(edgelist_df)

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
        edgelist_df["weight"] = cp.float32(1)
    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    num_nodes = max(edgelist_df['src'].max().compute(),
                    edgelist_df['dst'].max().compute())

    # running into issues with smaller partitions
    edgelist_df = edgelist_df.repartition(npartitions=edgelist_df.npartitions*2)

    num_nodes_dict = {'_N':num_nodes}

    gs = CuGraphStorage(num_nodes_dict=num_nodes_dict,  single_gpu=False)
    gs.add_edge_data(edgelist_df,   
                    node_col_names=['dst', 'src'],
                    canonical_etype=['_N', 'C', '_N'])
    return (gs, client, cluster)



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

    num_verts = G.total_number_of_nodes

    if batch_size > num_verts:
        num_start_verts = int(num_verts * 0.25)
    else:
        num_start_verts = batch_size
    
    srcs = G.graphstore.gdata.get_edge_data()['_SRC_']
    start_list = srcs.head(num_start_verts)
    assert len(start_list) == num_start_verts

    return {
        "seed_nodes": torch.as_tensor(start_list.values),
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
    else:
        (G, dask_client, dask_cluster) = create_mg_graph(graph_data)
    print(f"done creating graph, took {((time.perf_counter_ns() - st) / 1e9)}s")

    yield G

    if dask_client is not None:
        dask_client.shutdown()
        dask_cluster.close()



################################################################################
# Benchmarks
@pytest.mark.parametrize("batch_size", params.batch_sizes.values())
@pytest.mark.parametrize("fanout", [params.fanout_10_25, params.fanout_5_10_15])
@pytest.mark.parametrize(
    "with_replacement", [False], ids=lambda v: f"with_replacement={v}"
)
def bench_cugraph_dgl_uniform_neighbor_sample(
    benchmark, graph_objs, batch_size, fanout, with_replacement
):
    G = graph_objs
    uns_args = get_uniform_neighbor_sample_args(
        G, _seed, batch_size, fanout, with_replacement
    )

    # Reverse to match cugraph
    # DGL does from dst to src
    fanout_val = uns_args['fanout']
    fanout_val.reverse()
    sampler = dgl.dataloading.NeighborSampler(uns_args["fanout"])
    sampler_f = sampler.sample_blocks
    
    # Warmup
    _ = sampler_f(g=G, seed_nodes=uns_args["seed_nodes"])
    # print(f"\n{uns_args}")
    result_seed_nodes, output_nodes, blocks  = benchmark(
        sampler_f,
        g=G,
        seed_nodes=uns_args["seed_nodes"],
    )
    dt = str(result_seed_nodes.dtype)
    llen = len(result_seed_nodes)
    print(f"\nresult list len: {llen} , dtype={dt}")
