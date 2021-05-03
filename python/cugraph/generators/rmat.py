# Copyright (c) 2021, NVIDIA CORPORATION.
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

from dask.distributed import wait, default_client, Client
import dask_cudf
import cudf

from cugraph.generators import rmat_wrapper
from cugraph.comms import comms as Comms
import cugraph


def _ensure_args_rmat(scale,
                      num_edges,
                      a,
                      b,
                      c,
                      seed,
                      clip_and_flip,
                      scramble_vertex_ids,
                      create_using,
                      mg
):
    """
    Ensures the args passed in are usable for the rmat() API, raises the
    appropriate exception if incorrect, else returns None.
    """
    if mg and create_using is not cugraph.DiGraph:
        raise TypeError("Only cugraph.DiGraph can be used for multi-GPU R-MAT")
    if create_using not in [cugraph.Graph, cugraph.DiGraph]:
        raise TypeError("Only cugraph.Graph and cugraph.DiGraph are supported"
                        "types for 'create_using'")
    if not isinstance(scale, int):
        raise TypeError("'scale' must be an int")
    if (a+b+c > 1):
        raise ValueError("a + b + c should be non-negative and no larger than 1.0")
    if (clip_and_flip not in [True, False]):
        raise ValueError("'clip_and_flip' must be a bool")
    if (scramble_vertex_ids not in [True, False]):
        raise ValueError("'clip_and_flip' must be a bool")


def _ensure_args_multi_rmat(n_edgelists,
                            min_scale,
                            max_scale,
                            edge_factor,
                            size_distribution,
                            edge_distribution,
                            seed,
                            clip_and_flip,
                            scramble_vertex_ids
):
    """
    Ensures the args passed in are usable for the multi_rmat() API, raises the
    appropriate exception if incorrect, else returns None.
    """
    pass


def _sg_rmat(scale,
             num_edges,
             a,
             b,
             c,
             seed,
             clip_and_flip,
             scramble_vertex_ids,
             create_using=cugraph.DiGraph
):
    """
    FIXME: add docstring
    """
    df = rmat_wrapper.generate_rmat_edgelist(scale,
                                             num_edges,
                                             a,
                                             b,
                                             c,
                                             seed,
                                             clip_and_flip,
                                             scramble_vertex_ids)

    G = create_using()
    G.from_cudf_edgelist(df, source='src', destination='dst')

    return G


def call_rmat(sID,
              scale,
              num_edges_for_worker,
              a,
              b,
              c,
              unique_worker_seed,
              clip_and_flip,
              scramble_vertex_ids
):
    handle = Comms.get_handle(sID)

    return rmat_wrapper.generate_rmat_edgelist(scale,
                                               num_edges_for_worker,
                                               a,
                                               b,
                                               c,
                                               unique_worker_seed,
                                               clip_and_flip,
                                               scramble_vertex_ids,
                                               handle=handle
                                              )


def _mg_rmat(scale,
             num_edges,
             a,
             b,
             c,
             seed,
             clip_and_flip,
             scramble_vertex_ids,
             create_using=cugraph.DiGraph
):
    client = default_client()
    num_workers = len(client.scheduler_info()['workers'])

    num_edges_list = calc_num_edges_per_worker(num_workers, num_edges)

    futures = []
    for (i, worker_num_edges) in enumerate(num_edges_list):
        unique_worker_seed = seed + i
        future = client.submit(call_rmat,
                               Comms.get_session_id(),
                               scale,
                               worker_num_edges,
                               a,
                               b,
                               c,
                               unique_worker_seed,
                               clip_and_flip,
                               scramble_vertex_ids,
                              )
        futures.append(future)

    wait(futures)

    # Create the graph from the distributed dataframe(s), first by creating a
    # dask_cudf DataFrame, then by populating a Graph object with it (making it
    # a distributed graph)
    # FIXME: verify if this is correct!
    ddf = dask_cudf.from_cudf(futures[0].result(), npartitions=1)
    for f in futures[1:]:
        ddf = ddf.append(f.result())

    G = create_using()
    G.from_dask_cudf_edgelist(ddf, source="src", destination="dst")

    return G


def calc_num_edges_per_worker(num_workers, num_edges):
    """
    FIXME: add docstring
    """
    #48 and 10
    L= []
    w = num_edges//num_workers
    r = num_edges%num_workers
    for i in range (num_workers):
        if (i<r):
            L.append(w+1)
        else:
            L.append(w)
    return L


################################################################################

def rmat(scale,
         num_edges,
         a,
         b,
         c,
         seed,
         clip_and_flip,
         scramble_vertex_ids,
         create_using=cugraph.DiGraph,
         mg=False
):
    """
    Generate a Graph object using a Recursive MATrix (R-MAT) graph generation algorithm.

    Parameters
    ----------
    scale : int
    Scale factor to set the number of verties in the graph Vertex IDs have
    values in [0, V), where V = 1 << 'scale'

    num_edges : int
    Number of edges to generate

    a : float
    Probability of the first partition

    b : float
    Probability of the second partition

    c : float
    Probability of the thrid partition

    seed : int
    Seed value for the random number generator

    clip_and_flip : bool
    Flag controlling whether to generate edges only in the lower triangular part
    (including the diagonal) of the graph adjacency matrix (if set to 'true') or
    not (if set to 'false).

    scramble_vertex_ids : bool
    Flag controlling whether to scramble vertex ID bits (if set to `true`) or
    not (if set to `false`); scrambling vertx ID bits breaks correlation between
    vertex ID values and vertex degrees

    create_using : cugraph Graph type
    The graph type to construct containing the generated edges and vertices.
    Default is cugraph.DiGraph.  NOTE: only the cugraph.DiGraph type is
    supported for multi-GPU

    mg : bool
    If True, R-MAT generation occurs across multiple GPUs. If False, only a
    single GPU is used.  Default is False (single-GPU)

    Returns
    -------
    instance of cugraph.Graph
    """
    _ensure_args_rmat(scale, num_edges, a, b, c, seed, clip_and_flip,
                      scramble_vertex_ids, create_using, mg)

    if mg:
        return _mg_rmat(scale, num_edges, a, b, c, seed, clip_and_flip,
                        scramble_vertex_ids, create_using)
    else:
        return _sg_rmat(scale, num_edges, a, b, c, seed, clip_and_flip,
                        scramble_vertex_ids, create_using)


def multi_rmat(
    n_edgelists,
    min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids
):
    """
    Generate multiple Graph objects using a Recursive MATrix (R-MAT) graph generation algorithm.

    Parameters
    ----------
    n_edgelists : int
    Number of edge lists (graphs) to generate

    min_scale : int
    Scale factor to set the minimum number of vertices in the graph

    max_scale : int
    Scale factor to set the maximum number of vertices in the graph

    edge_factor : int
    Average number of edges per vertex to generate

    size_distribution :
    Distribution of the graph sizes, impacts the scale parameter of the R-MAT
    generator

    edge_distribution :
    Edges distribution for each graph, impacts how R-MAT parameters a,b,c,d, are
    set

    seed : int
    Seed value for the random number generator

    clip_and_flip : bool
    Flag controlling whether to generate edges only in the lower triangular part
    (including the diagonal) of the graph adjacency matrix (if set to 'true') or
    not (if set to 'false')

    scramble_vertex_ids : bool
    Flag controlling whether to scramble vertex ID bits (if set to `true`) or
    not (if set to `false`); scrambling vertx ID bits breaks correlation between
    vertex ID values and vertex degrees

    Returns
    -------
    list of cugraph.Graph instances
    """
    _ensure_args_multi_rmat(n_edgelists, min_scale, max_scale, edge_factor,
                            size_distribution, edge_distribution, seed,
                            clip_and_flip, scramble_vertex_ids)

    dfs = rmat_wrapper.generate_rmat_edgelists(n_edgelists, min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids)
    list_G = []

    for df in dfs:
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source='src', destination='dst')
        list_G.append(G)


    return list_G
