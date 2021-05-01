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

from cugraph.generators import rmat_wrapper
import cugraph.comms.comms as Comms
import cugraph


def _ensure_args_rmat(scale,
                      a,
                      b,
                      c,
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
        raise TypeError("Only cugraph.DiGraph can be used for multi-GPU RMAT")
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
    #client = default_client()
    client = Client() #change this
    num_workers = len(client.scheduler_info()['workers'])

    list_job = calc_num_edges_per_worker(num_workers, num_edges)

    #edges = get_distributed_data() #call function to distribute the edge generation
    #78 10

    L=[client.submit(graph_generator,
                               scale,
                               n_edges,
                               a,
                               b,
                               c,
                               seed,
                               clip_and_flip,
                               scramble_vertex_ids) for seed, n_edges in enumerate(list_job)]


    #client.gather(L)

    # FIXME: need to return a Graph
    return L


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
    """Generate a Graph object using a Recursive MATrix (R-MAT) graph generation algorithm.

    Parameters
    ----------
    scale : int
            Scale factor to set the number of verties in the graph
            Vertex IDs have values in [0, V), where V = 1 << 'scale'

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
            (including the diagonal) of the graph adjacency matrix (if set to 'true')
            or not (if set to 'false).

    scramble_vertex_ids : bool
            Flag controlling whether to scramble vertex ID bits (if set to `true`)
            or not (if set to `false`); scrambling vertx ID bits breaks correlation between
            vertex ID values and vertex degrees

    create_using : cugraph Graph type
            The graph type to construct containing the generated edges and
            vertices.  Default is cugraph.DiGraph.
            NOTE: only the cugraph.DiGraph type is supported for multi-GPU

    mg : bool
            If True, RMAT generation occurs across multiple GPUs. If False, only
            a single GPU is used.  Default is False (single-GPU)

    Returns
    -------
    instance of cugraph.Graph

    """
    _ensure_args_rmat(scale, a, b, c, clip_and_flip,
                      scramble_vertex_ids, create_using, mg)

    if mg:
        return _sg_rmat(scale, a, b, c, clip_and_flip,
                        scramble_vertex_ids, create_using)
    else:
        return _mg_rmat(scale, a, b, c, clip_and_flip,
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
        Distribution of the graph sizes, impacts the scale parameter of the
        R-MAT generator

    edge_distribution :
        Edges distribution for each graph, impacts how R-MAT parameters a,b,c,d,
        are set

    seed : int
        Seed value for the random number generator

    clip_and_flip : bool
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix (if set to 'true')
        or not (if set to 'false')

    scramble_vertex_ids : bool
        Flag controlling whether to scramble vertex ID bits (if set to `true`)
        or not (if set to `false`); scrambling vertx ID bits breaks correlation between vertex
        ID values and vertex degrees

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
