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

from cugraph.dask.comms import comms as Comms
from dask_cuda import LocalCUDACluster
from dask.distributed import Client

import os
import time
import pytest
import numpy as np
import cupy as cp
#import torch
from cugraph.testing.mg_utils import start_dask_client, stop_dask_client
import cudf
import dask_cudf
import rmm
import cugraph

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
from cugraph.experimental import datasets, PropertyGraph, MGPropertyGraph
from cugraph.dask import uniform_neighbor_sample as uniform_neighbor_sample_mg

from cugraph_pyg.sampler import CuGraphSampler
from cugraph_pyg.data import to_pyg

from cugraph_benchmarking import params

_seed = 42


def create_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """

    rmm.reinitialize(pool_allocator=True, initial_pool_size=10e9, maximum_pool_size=28e9)

    pG = PropertyGraph()

    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        
        vertex_df = cudf.concat(
            [edgelist_df['src'], edgelist_df['dst']]
        ).unique()
        vertex_df.name = 'vtx'

        pG.add_vertex_data(vertex_df.to_frame(), vertex_col_name='vtx', type_name='vt1')

        pG.add_edge_data(
            edgelist_df, vertex_col_names=['src','dst'],
            type_name='et1',
            property_columns=["wgt"],
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
        edgelist_df['eid'] = edgelist_df.index.to_series().astype('int64')
        edgelist_df['src'] = edgelist_df.src.astype('int64')
        edgelist_df['dst'] = edgelist_df.src.astype('int64')

        vertex_df = cudf.concat(
            [edgelist_df['src'], edgelist_df['dst']]
        ).unique()
        vertex_df.name = 'vtx'

        pG.add_vertex_data(vertex_df.to_frame(), vertex_col_name='vtx', type_name='vt1')

        pG.add_edge_data(
            edgelist_df,
            vertex_col_names=['src','dst'],
            type_name='et1',
            edge_id_col_name='eid',
            property_columns=['weight']
        )

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return pG


def create_mg_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """
    ## Reserving GPU 0 for client(trainer/service project)
    start = 0
    n_devices = os.getenv('DASK_NUM_WORKERS', 4)
    n_devices = int(n_devices)

    visible_devices = ','.join([str(i) for i in range(1 + start, n_devices + 1 + start)])

    cluster = LocalCUDACluster(protocol='ucx', rmm_pool_size='31GB', CUDA_VISIBLE_DEVICES=visible_devices)
    client = Client(cluster)
    Comms.initialize(p2p=True)
    rmm.reinitialize(pool_allocator=True)

    
    pG = MGPropertyGraph()

    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()
        # FIXME: edgelist_df should have column names that match the defaults
        # for G.from_cudf_edgelist()
        edgelist_df = dask_cudf.from_cudf(edgelist_df, npartitions=n_devices)
        #edgelist_df = edgelist_df.repartition(npartitions=edgelist_df.npartitions*2).persist()
        edgelist_df.compute()

        vertex_df = dask_cudf.concat(
            [edgelist_df['src'], edgelist_df['dst']]
        ).unique()
        vertex_df.name = 'vtx'

        pG.add_vertex_data(vertex_df.to_frame(), vertex_col_name='vtx', type_name='vt1')

        pG.add_edge_data(
            edgelist_df,
            vertex_col_names=['src','dst'],
            type_name='et1',
            property_columns=['wgt']
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
        edgelist_df['src'] = edgelist_df.src.astype('int64')
        edgelist_df['dst'] = edgelist_df.dst.astype('int64')

        vertex_df = dask_cudf.concat(
            [edgelist_df['src'], edgelist_df['dst']]
        ).unique()
        vertex_df.name = 'vtx'
    
        vertex_df = vertex_df.to_frame().reset_index().rename(columns={'index':'vid'})

        edgelist_df = edgelist_df.merge(
            vertex_df, how='left', left_on='src', right_on='vtx'
        ).drop(
            ['src', 'vtx'], axis=1
        ).rename(
            columns={'vid':'src'}
        ).merge(
            vertex_df, how='left', left_on='dst', right_on='vtx'
        ).drop(
            ['dst', 'vtx'], axis=1
        ).rename(
            columns={'vid':'dst'}
        ).reset_index(drop=True).repartition(npartitions=n_devices*4)
        
        edgelist_df["weight"] = cp.float32(1)
        edgelist_df['eid'] = 1
        edgelist_df['eid'] = edgelist_df['eid'].cumsum() - 1
        edgelist_df = edgelist_df.persist()

        pG.add_vertex_data(vertex_df, vertex_col_name='vid', type_name='vt1')

        pG.add_edge_data(
            edgelist_df,
            vertex_col_names=['src','dst'],
            type_name='et1',
            edge_id_col_name='eid',
            property_columns=['weight']
        )

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return (pG, client, cluster)


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
    num_verts = G.get_num_vertices()

    if batch_size > num_verts:
        num_start_verts = int(num_verts * 0.25)
    else:
        num_start_verts = batch_size

    # Create the list of starting vertices by picking num_start_verts random
    # ints between 0 and num_verts, then map those to actual vertex IDs.  Since
    # the randomly-chosen IDs may not map to actual IDs, keep trying until
    # num_start_verts have been picked, or max_tries is reached.

    #start_list = cp.random.randint(0, num_verts, num_start_verts, dtype='int64')
    import torch
    start_list = torch.randint(0, num_verts, (num_start_verts,), dtype=torch.int64)
    #print('start len:', len(start_list))
    #print(G.get_vertex_data(vertex_ids=start_list).compute())

    return {
        #"start_list": torch.tensor(start_list, dtype=torch.int32).cuda(),
        'start_list': start_list,
        "fanout": fanout,
        "with_replacement": with_replacement,
    }



def get_sampler(data, num_neighbors, with_replacement, directed, edge_types):
    sampler = CuGraphSampler(
        data, 
        method='uniform_neighbor',        
        replace=with_replacement,
        directed=directed,
        edge_types=edge_types,
        num_neighbors=num_neighbors
    )

    return sampler

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
        H = cugraph.MultiGraph(directed=True)
        H.from_dask_cudf_edgelist(G._edge_prop_dataframe, source=G.src_col_name, destination=G.dst_col_name, edge_attr='weight')
        raise ValueError('got subgraph!')

    # G.renumber_vertices_by_type()
    # G.renumber_edges_by_type()
    # Renumber edges by type is not needed.
    # Edges are already contiguously numbered for a single type.

    print(f"done creating graph, took {((time.perf_counter_ns() - st) / 1e9)}s")

    data = to_pyg(G, renumber_graph=False, backend='cupy')
    # prefetch the subgraph
    #data[0]._subgraph(['et1'])
    #H = cugraph.MultiGraph(directed=True)
    #H.from_dask_cudf_edgelist(G._edge_prop_dataframe, source=G.src_col_name, destination=G.dst_col_name, edge_attr='weight')
    #data[0]._EXPERIMENTAL__CuGraphStore__subgraphs[('et1',)] = H
    #raise ValueError(f'got subgraph!')
    yield (G, data)

    if dask_client is not None:
        stop_dask_client(dask_client, dask_cluster)

################################################################################
# Benchmarks
@pytest.mark.parametrize("batch_size", params.batch_sizes.values())
#@pytest.mark.parametrize('batch_size', [100, 500, 1000])
#@pytest.mark.parametrize("fanout", [params.fanout_10_25, params.fanout_5_10_15])
@pytest.mark.parametrize("fanout", [params.fanout_10_25])
@pytest.mark.parametrize(
    "with_replacement", [False], ids=lambda v: f"with_replacement={v}"
)
def bench_cugraph_uniform_neighbor_sample(
    gpubenchmark, graph_objs, batch_size, fanout, with_replacement
):
    G, data = graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, _seed, batch_size, fanout, with_replacement
    )

    sampler = get_sampler(
        data,
        num_neighbors=uns_args['fanout'],
        with_replacement=uns_args["with_replacement"],
        directed=True,
        edge_types=None,
    )
    uns_func = lambda ix : sampler.sample_from_nodes((None, ix, None))

    # print(f"\n{uns_args}")
    
    # import pdb;pdb.set_trace()

    # uns_func(uns_args['start_list'])

    result = gpubenchmark(
        uns_func,
        ix=uns_args["start_list"],
    )
    
    
    # noi_index, row_dict, col_dict, _ = result['out']

    # dtmap = {torch.int32: 32 // 8, torch.int64: 64 // 8}
    # bytes = 0
    # llen = 0
    # for d in noi_index, row_dict, col_dict:
    #    for x in d.values():
    #        llen += len(x)
    #        bytes += len(x) * dtmap[x.dtype]
    # print(f"\nresult list len: {llen} (x3), total bytes={bytes}")
