# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import numpy as np
import dask_cudf
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from cugraph.structure.symmetrize import symmetrize_ddf
from cugraph.dask.common.mg_utils import get_visible_devices
from dask_cuda.initialize import initialize
import cudf

import cugraph
from cugraph.dask.comms import comms as Comms
from cugraph.generators import rmat
import tempfile

import rmm


def generate_edgelist(scale,
                      edgefactor,
                      seed=None,
                      unweighted=False,
):
    """
    Returns a dask_cudf DataFrame created using the R-MAT graph generator.

    The resulting graph is weighted with random values of a uniform distribution
    from the interval [0, 1)

    scale is used to determine the number of vertices to be generated (num_verts
    = 2^scale), which is also used to determine the data type for the vertex ID
    values in the DataFrame.

    edgefactor determies the number of edges (num_edges = num_edges*edgefactor)

    seed, if specified, will be used as the seed to the RNG.

    unweighted determines if the resulting edgelist will have randomly-generated
    weightes ranging in value between [0, 1). If True, an edgelist with only 2
    columns is returned.
    """
    ddf = rmat(
        scale,
        (2**scale)*edgefactor,
        0.57,  # from Graph500
        0.19,  # from Graph500
        0.19,  # from Graph500
        seed or 42,
        clip_and_flip=False,
        scramble_vertex_ids=True,
        create_using=None,  # return edgelist instead of Graph instance
        mg=True
    )
    if not unweighted:
        rng = np.random.default_rng(seed)
        ddf["weight"] = ddf.map_partitions(lambda df: rng.random(size=len(df)))
    return ddf


def read_csv(input_csv_file, scale):
    """
    Returns a dask_cudf DataFrame from reading input_csv_file.

    All input CSV files should be weighted with random values of a uniform
    distribution from the interval [0, 1) in order to best simulate the output
    of a Graph500-compliant graph generator.

    scale is used to determine the data type for the vertex ID values in the
    DataFrame. (num verts = 2^scale), which is used to determine
    """
    vertex_t = "int32" if scale <= 32 else "int64"
    dtypes = [vertex_t, vertex_t, "float32"]
    names=["src", "dst", "weight"],

    chunksize = cugraph.dask.get_chunksize(input_csv_file)
    return dask_cudf.read_csv(input_csv_file,
                              chunksize=chunksize,
                              delimiter=" ",
                              #names=names,
                              dtype=dtypes,
                              header=None,
    )


################################################################################
# Benchmarked functions
#
# The "benchmark_name" attr is used by the benchmark infra for reporting and is
# set to assign more meaningful names to be displayed in reports.

def construct_graph(dask_dataframe, symmetric=False):
    """
    dask_dataframe contains weighted and undirected edges with self
    loops. Multiple edges will likely be present as well.  The returned Graph
    object must be symmetrized and have self loops removed.
    """

    if symmetric:
        G = cugraph.Graph(directed=False)
    else:
        G = cugraph.Graph(directed=True)

    if len(dask_dataframe.columns) > 2:
        if symmetric: #symmetrize dask dataframe
            dask_dataframe = symmetrize_ddf(
                dask_dataframe, 'src', 'dst', 'weight')

        G.from_dask_cudf_edgelist(
            dask_dataframe, source="src", destination="dst", edge_attr="weight")
        #G.from_dask_cudf_edgelist(
        #    dask_dataframe, source="0", destination="1", edge_attr="2")
    else:
        if symmetric: #symmetrize dask dataframe
            dask_dataframe = symmetrize_ddf(dask_dataframe, 'src', 'dst')
        G.from_dask_cudf_edgelist(
            dask_dataframe, source="src", destination="dst")

    return G

construct_graph.benchmark_name = "from_dask_cudf_edgelist"


def bfs(G, start):
    return cugraph.dask.bfs(
        G, start=start, return_distances=True, check_start=False)


def sssp(G, start):
    return cugraph.dask.sssp(G, source=start, check_source=False)


def wcc(G):
    return cugraph.dask.weakly_connected_components(G)


def louvain(G):
    return cugraph.dask.louvain(G)


def pagerank(G):
    return cugraph.dask.pagerank(G)


def katz(G, alpha=None):
    print(alpha)
    return cugraph.dask.katz_centrality(G, alpha)

def hits(G):
    return cugraph.dask.hits(G)

def uniform_neighbor_sample(G, start_list=None, fanout_vals=None):
    # convert list to cudf.Series
    start_list = cudf.Series(start_list, dtype="int32")  
    return cugraph.dask.uniform_neighbor_sample(
        G, start_list=start_list, fanout_vals=fanout_vals)

def triangle_count(G):
    return cugraph.dask.triangle_count(G)

def eigenvector_centrality(G):
    return cugraph.dask.eigenvector_centrality(G)

################################################################################
# Session-wide setup and teardown

def setup(dask_scheduler_file=None, rmm_pool_size=None):
    if dask_scheduler_file:
        cluster = None
        # Env var UCX_MAX_RNDV_RAILS=1 must be set too.
        initialize(enable_tcp_over_ucx=True,
                   enable_nvlink=True,
                   enable_infiniband=False,
                   enable_rdmacm=False,
                   #net_devices="mlx5_0:1",
                  )
        client = Client(scheduler_file=dask_scheduler_file)

    else:
        tempdir_object = tempfile.TemporaryDirectory()
        cluster = LocalCUDACluster(local_directory=tempdir_object.name, rmm_pool_size=rmm_pool_size)
        client = Client(cluster)
        # add the obj to the client so it doesn't get deleted until
        # the 'client' obj gets cleaned up
        client.tempdir_object = tempdir_object
        client.wait_for_workers(len(get_visible_devices()))

    Comms.initialize(p2p=True)
    return (client, cluster)


def teardown(client, cluster=None):
    Comms.destroy()
    # Shutdown the connected scheduler and workers
    # therefore we will no longer rely on killing the dask cluster ID
    # for MNMG runs
    client.shutdown()
    if cluster:
        cluster.close()
