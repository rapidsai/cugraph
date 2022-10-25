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

from dask.distributed import default_client
import dask_cudf

from cugraph.generators import rmat_wrapper
from cugraph.dask.comms import comms as Comms
import cugraph

_graph_types = [cugraph.Graph, cugraph.MultiGraph]


def _ensure_args_rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    create_using,
    mg,
):
    """
    Ensures the args passed in are usable for the rmat() API, raises the
    appropriate exception if incorrect, else returns None.
    """
    if create_using is not None:
        if isinstance(create_using, cugraph.Graph):
            directed = create_using.is_directed()
            if mg and not directed:
                raise TypeError(
                    "Only directed cugraph.Graph and None "
                    "are supported types for `create_using` "
                    "and `directed` for multi-GPU R-MAT"
                )
        elif create_using not in _graph_types:
            raise TypeError(
                "create_using must be a cugraph.Graph "
                "(or subclass) type or instance, got: "
                f"{type(create_using)}"
            )
    if not isinstance(scale, int):
        raise TypeError("'scale' must be an int")
    if not isinstance(num_edges, int):
        raise TypeError("'num_edges' must be an int")
    if a + b + c > 1:
        raise ValueError("a + b + c should be non-negative and no larger than 1.0")
    if clip_and_flip not in [True, False]:
        raise ValueError("'clip_and_flip' must be a bool")
    if scramble_vertex_ids not in [True, False]:
        raise ValueError("'scramble_vertex_ids' must be a bool")
    if not isinstance(seed, int):
        raise TypeError("'seed' must be an int")


def _ensure_args_multi_rmat(
    n_edgelists,
    min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
):
    """
    Ensures the args passed in are usable for the multi_rmat() API, raises the
    appropriate exception if incorrect, else returns None.

    """
    if not isinstance(n_edgelists, int):
        raise TypeError("'n_edgelists' must be an int")
    if not isinstance(min_scale, int):
        raise TypeError("'min_scale' must be an int")
    if not isinstance(max_scale, int):
        raise TypeError("'max_scale' must be an int")
    if not isinstance(edge_factor, int):
        raise TypeError("'edge_factor' must be an int")
    if size_distribution not in [0, 1]:
        raise TypeError("'size_distribution' must be either 0 or 1")
    if edge_distribution not in [0, 1]:
        raise TypeError("'edge_distribution' must be either 0 or 1")
    if clip_and_flip not in [True, False]:
        raise ValueError("'clip_and_flip' must be a bool")
    if scramble_vertex_ids not in [True, False]:
        raise ValueError("'scramble_vertex_ids' must be a bool")
    if not isinstance(seed, int):
        raise TypeError("'seed' must be an int")


def _sg_rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    create_using=cugraph.Graph,
):
    """
    Calls RMAT on a single GPU and uses the resulting cuDF DataFrame
    to initialize and return a cugraph Graph object specified with
    create_using. If create_using is None, returns the edgelist df as-is.
    """
    df = rmat_wrapper.generate_rmat_edgelist(
        scale, num_edges, a, b, c, seed, clip_and_flip, scramble_vertex_ids
    )
    if create_using is None:
        return df

    if isinstance(create_using, cugraph.Graph):
        attrs = {"directed": create_using.is_directed()}
        G = type(create_using)(**attrs)
    elif create_using in _graph_types:
        G = create_using()
    else:
        raise TypeError(
            "create_using must be a cugraph.Graph "
            "(or subclass) type or instance, got: "
            f"{type(create_using)}"
        )
    G.from_cudf_edgelist(df, source="src", destination="dst", renumber=False)

    return G


def _mg_rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    create_using=cugraph.Graph,
):
    """
    Calls RMAT on multiple GPUs and uses the resulting Dask cuDF DataFrame to
    initialize and return a cugraph Graph object specified with create_using.
    If create_using is None, returns the Dask DataFrame edgelist as-is.

    seed is used as the initial seed for the first worker used (worker 0), then
    each subsequent worker will receive seed+<worker num> as the seed value.
    """
    client = default_client()
    worker_list = list(client.scheduler_info()["workers"].keys())
    num_workers = len(worker_list)
    num_edges_list = _calc_num_edges_per_worker(num_workers, num_edges)
    futures = []
    for (i, worker_num_edges) in enumerate(num_edges_list):
        unique_worker_seed = seed + i
        future = client.submit(
            _call_rmat,
            Comms.get_session_id(),
            scale,
            worker_num_edges,
            a,
            b,
            c,
            unique_worker_seed,
            clip_and_flip,
            scramble_vertex_ids,
            workers=worker_list[i],
        )
        futures.append(future)

    ddf = dask_cudf.from_delayed(futures)

    if create_using is None:
        return ddf

    if isinstance(create_using, cugraph.Graph):
        attrs = {"directed": create_using.is_directed()}
        G = type(create_using)(**attrs)
    elif create_using in _graph_types:
        G = create_using()
    else:
        raise TypeError(
            "create_using must be a cugraph.Graph "
            "(or subclass) type or instance, got: "
            f"{type(create_using)}"
        )
    G.from_dask_cudf_edgelist(ddf, source="src", destination="dst")

    return G


def _call_rmat(
    sID,
    scale,
    num_edges_for_worker,
    a,
    b,
    c,
    unique_worker_seed,
    clip_and_flip,
    scramble_vertex_ids,
):
    """
    Callable passed to dask client.submit calls that extracts the individual
    worker handle based on the dask session ID
    """
    handle = Comms.get_handle(sID)

    return rmat_wrapper.generate_rmat_edgelist(
        scale,
        num_edges_for_worker,
        a,
        b,
        c,
        unique_worker_seed,
        clip_and_flip,
        scramble_vertex_ids,
        handle=handle,
    )


def _calc_num_edges_per_worker(num_workers, num_edges):
    """
    Returns a list of length num_workers with the individual number of edges
    each worker should generate. The sum of all edges in the list is num_edges.
    """
    L = []
    w = num_edges // num_workers
    r = num_edges % num_workers
    for i in range(num_workers):
        if i < r:
            L.append(w + 1)
        else:
            L.append(w)
    return L


###############################################################################


def rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    create_using=cugraph.Graph,
    mg=False,
):
    """
    Generate a Graph object using a Recursive MATrix (R-MAT) graph generation
    algorithm.

    Parameters
    ----------
    scale : int
        Scale factor to set the number of vertices in the graph Vertex IDs have
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
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix
        (if set to 'true') or not (if set to 'false).

    scramble_vertex_ids : bool
        Flag controlling whether to scramble vertex ID bits (if set to `true`)
        or not (if set to `false`); scrambling vertex ID bits breaks
        correlation between vertex ID values and vertex degrees.

    create_using : cugraph Graph type or None The graph type to construct
        containing the generated edges and vertices.  If None is specified, the
        edgelist cuDF DataFrame (or dask_cudf DataFrame for MG) is returned
        as-is. This is useful for benchmarking Graph construction steps that
        require raw data that includes potential self-loops, isolated vertices,
        and duplicated edges.  Default is cugraph.Graph.

    mg : bool, optional (default=False)
        If True, R-MAT generation occurs across multiple GPUs. If False, only a
        single GPU is used.  Default is False (single-GPU)

    Returns
    -------
    instance of cugraph.Graph

    Examples
    --------
    >>> import cugraph
    >>> from cugraph.generators import rmat
    >>> df = rmat(
    ...    scale,
    ...    (2**scale)*edgefactor,
    ...    0.57,
    ...    0.19,
    ...    0.19,
    ...    seed or 42,
    ...    clip_and_flip=False,
    ...    scramble_vertex_ids=True,
    ...    create_using=None,  # return edgelist instead of Graph instance
    ...    mg=False
    ... )

    """

    _ensure_args_rmat(
        scale,
        num_edges,
        a,
        b,
        c,
        seed,
        clip_and_flip,
        scramble_vertex_ids,
        create_using,
        mg,
    )

    if mg:
        return _mg_rmat(
            scale,
            num_edges,
            a,
            b,
            c,
            seed,
            clip_and_flip,
            scramble_vertex_ids,
            create_using,
        )
    else:
        return _sg_rmat(
            scale,
            num_edges,
            a,
            b,
            c,
            seed,
            clip_and_flip,
            scramble_vertex_ids,
            create_using,
        )


def multi_rmat(
    n_edgelists,
    min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
):
    """
    Generate multiple Graph objects using a Recursive MATrix (R-MAT) graph
    generation algorithm.

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

    size_distribution : int
        Distribution of the graph sizes, impacts the scale parameter of the
        R-MAT generator.
        '0' for POWER_LAW distribution and '1' for UNIFORM distribution

    edge_distribution : int
        Edges distribution for each graph, impacts how R-MAT parameters
        a,b,c,d, are set.
        '0' for POWER_LAW distribution and '1' for UNIFORM distribution

    seed : int
        Seed value for the random number generator

    clip_and_flip : bool
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix
        (if set to 'true') or not (if set to 'false')

    scramble_vertex_ids : bool
        Flag controlling whether to scramble vertex ID bits (if set to 'true')
        or not (if set to 'false'); scrambling vertx ID bits breaks correlation
        between vertex ID values and vertex degrees

    Returns
    -------
    list of cugraph.Graph instances
    """
    _ensure_args_multi_rmat(
        n_edgelists,
        min_scale,
        max_scale,
        edge_factor,
        size_distribution,
        edge_distribution,
        seed,
        clip_and_flip,
        scramble_vertex_ids,
    )

    dfs = rmat_wrapper.generate_rmat_edgelists(
        n_edgelists,
        min_scale,
        max_scale,
        edge_factor,
        size_distribution,
        edge_distribution,
        seed,
        clip_and_flip,
        scramble_vertex_ids,
    )
    list_G = []

    for df in dfs:
        G = cugraph.Graph()
        G.from_cudf_edgelist(df, source="src", destination="dst")
        list_G.append(G)

    return list_G
