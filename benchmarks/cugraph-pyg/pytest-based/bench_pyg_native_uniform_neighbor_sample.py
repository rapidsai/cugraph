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
import torch
import cudf

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


from cugraph.generators import rmat
from cugraph.experimental import datasets
from torch_geometric.data import HeteroData

from cugraph_benchmarking import params

_seed = 42


def create_graph(graph_data):
    """
    Create a graph instance based on the data to be loaded/generated.
    """

    # Assume strings are names of datasets in the datasets package
    if isinstance(graph_data, str):
        ds = getattr(datasets, graph_data)
        edgelist_df = ds.get_edgelist()

        ei = torch.stack([
            torch.from_dlpack(edgelist_df['src'].to_dlpack()).to(torch.long),
            torch.from_dlpack(edgelist_df['dst'].to_dlpack()).to(torch.long)
        ])
        
        vertex_df = cudf.concat(
            [edgelist_df['src'], edgelist_df['dst']]
        ).unique()
        vertex_df.name = 'vtx'

        data = HeteroData().to('cuda:0')

        data['vtx'].x = torch.from_dlpack(vertex_df.to_dlpack()).to(torch.long)
        data[('vtx', 'et1', 'vtx')].edge_index = ei
        return data
    
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

        
        vertex_df = cudf.concat(
            [edgelist_df['src'], edgelist_df['dst']]
        ).unique()
        vertex_df.name = 'vtx'
        v_idx = cudf.Series(cp.arange(len(vertex_df)), index=vertex_df)

        edgelist_df = cudf.DataFrame({
            'src':v_idx.loc[edgelist_df['src'].to_cupy()].to_cupy(),
            'dst':v_idx.loc[edgelist_df['dst'].to_cupy()].to_cupy(),
        })

        edgelist_df["weight"] = cp.float32(1)
        ei = torch.stack([
            torch.from_dlpack(edgelist_df['src'].to_dlpack()).to(torch.long),
            torch.from_dlpack(edgelist_df['dst'].to_dlpack()).to(torch.long)
        ])

        with open('/work/testing/blah.file', 'w') as f:
            f.write('vertex ids:\n')
            f.write(str(edgelist_df.src.min()) + '\t' + str(edgelist_df.src.max()) + '\n')
            f.write(str(edgelist_df.dst.min()) + '\t' + str(edgelist_df.dst.max()) + '\n')
            f.write(str(len(vertex_df)) + '\n')


        data = HeteroData().to('cuda:0')

        data[('vtx', 'et1', 'vtx')].edge_index = ei
        data['vtx'].x = torch.from_dlpack(vertex_df.to_dlpack()).to(torch.float64)

    else:
        raise TypeError(f"graph_data can only be str or dict, got {type(graph_data)}")

    return data


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
    num_verts = G.num_nodes

    if batch_size > num_verts:
        num_start_verts = int(num_verts * 0.25)
    else:
        num_start_verts = batch_size

    # Create the list of starting vertices by picking num_start_verts random
    # ints between 0 and num_verts, then map those to actual vertex IDs.  Since
    # the randomly-chosen IDs may not map to actual IDs, keep trying until
    # num_start_verts have been picked, or max_tries is reached.
    
    #G.renumber_edges_by_type()
    #G.renumber_vertices_by_type()

    start_list_set = set()
    max_tries = 10000
    try_num = 0
    internal_vertex_ids_start_list = rng.choice(
        num_verts, size=num_start_verts, replace=False
    )
    internal_vertex_ids_start_list = torch.tensor(internal_vertex_ids_start_list).cuda()
    
    print(G.get_all_edge_attrs())
    start_list = G.get_edge_store('vtx','et1','vtx').edge_index.flatten().unique()[internal_vertex_ids_start_list]

    return {
        "start_list": start_list.to(torch.long),
        "fanout": fanout,
        "with_replacement": with_replacement,
    }



def get_sampler(data, num_neighbors, with_replacement, directed, edge_types):
    from torch_geometric.sampler.neighbor_sampler import NeighborSampler
    sampler = NeighborSampler(
        (data, data),
        replace=with_replacement,
        directed=directed,
        num_neighbors=num_neighbors,
        input_type='vtx'
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
    G = create_graph(graph_data)
    print(f"done creating graph, took {((time.perf_counter_ns() - st) / 1e9)}s")

    yield G

################################################################################
# Benchmarks
@pytest.mark.parametrize("batch_size", params.batch_sizes.values())
#@pytest.mark.parametrize("fanout", [params.fanout_10_25, params.fanout_5_10_15])
@pytest.mark.parametrize("fanout", [params.fanout_10_25])
@pytest.mark.parametrize(
    "with_replacement", [False], ids=lambda v: f"with_replacement={v}"
)
def bench_cugraph_uniform_neighbor_sample(
    gpubenchmark, graph_objs, batch_size, fanout, with_replacement
):
    G = graph_objs

    uns_args = get_uniform_neighbor_sample_args(
        G, _seed, batch_size, fanout, with_replacement
    )

    sampler = get_sampler(
        G,
        num_neighbors=uns_args['fanout'],
        with_replacement=uns_args["with_replacement"],
        directed=True,
        edge_types=None,
    )
    uns_func = lambda ix : sampler.sample_from_nodes((None, ix.cpu(), None))

    # print(f"\n{uns_args}")
    
    result = gpubenchmark(
        uns_func,
        ix=uns_args["start_list"],
    )
    
    print(result)


