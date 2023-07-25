# Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

from dask.distributed import default_client, wait
import dask_cudf

from cugraph.dask.comms import comms as Comms
import cudf
import numpy as np
import cupy as cp
import cugraph
from pylibcugraph import generate_rmat_edgelist as pylibcugraph_generate_rmat_edgelist
from pylibcugraph import generate_rmat_edgelists as pylibcugraph_generate_rmat_edgelists
from pylibcugraph import ResourceHandle

_graph_types = [cugraph.Graph, cugraph.MultiGraph]


def _ensure_args_rmat(
    scale=None,
    num_edges=None,
    a=None,
    b=None,
    c=None,
    n_edgelists=None,
    min_scale=None,
    max_scale=None,
    edge_factor=None,
    size_distribution=None,
    edge_distribution=None,
    seed=None,
    clip_and_flip=None,
    scramble_vertex_ids=None,
    include_edge_weights=None,
    minimum_weight=None,
    maximum_weight=None,
    dtype=None,
    include_edge_ids=None,
    include_edge_types=None,
    min_edge_type_value=None,
    max_edge_type_value=None,
    create_using=None,
    mg=None,
    multi_rmat=False,
):
    """
    Ensures the args passed in are usable for the rmat() or multi rmat() API,
    raises the appropriate exception if incorrect, else returns None.
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

    if not isinstance(seed, int):
        raise TypeError("'seed' must be an int")
    if include_edge_weights:
        if not isinstance(include_edge_weights, bool):
            raise TypeError("'include_edge_weights' must be a bool")
        if maximum_weight is None or minimum_weight is None:
            raise ValueError(
                "'maximum_weight' and 'minimum_weight' must not be 'None' "
                "if 'include_edge_weights' is True"
            )
        if dtype not in [
            np.float32,
            np.float64,
            cp.float32,
            cp.float64,
            "float32",
            "float64",
        ]:
            raise ValueError(
                "dtype must be either numpy or cupy 'float32' or 'float64' if "
                "'include_edge_weights' is True."
            )
    if include_edge_ids:
        if not isinstance(include_edge_ids, bool):
            raise TypeError("'include_edge_ids' must be a bool")
    if include_edge_types:
        if not isinstance(include_edge_types, bool):
            raise TypeError("'include_edge_types' must be a bool")
        if min_edge_type_value is None and max_edge_type_value is None:
            raise ValueError(
                "'min_edge_type_value' and 'max_edge_type_value' must not be 'None' "
                "if 'include_edge_types' is True"
            )

    if multi_rmat:
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
    else:
        if not isinstance(scale, int):
            raise TypeError("'scale' must be an int")
        if not isinstance(num_edges, int):
            raise TypeError("'num_edges' must be an int")
        if a + b + c > 1:
            raise ValueError("a + b + c should be non-negative and no larger than 1.0")
        if not isinstance(clip_and_flip, bool):
            raise TypeError("'clip_and_flip' must be a bool")
        if not isinstance(scramble_vertex_ids, bool):
            raise TypeError("'scramble_vertex_ids' must be a bool")


def _sg_rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    include_edge_weights,
    minimum_weight,
    maximum_weight,
    dtype,
    include_edge_ids,
    include_edge_types,
    min_edge_type_value,
    max_edge_type_value,
    create_using=cugraph.Graph,
):
    """
    Calls RMAT on a single GPU and uses the resulting cuDF DataFrame
    to initialize and return a cugraph Graph object specified with
    create_using. If create_using is None, returns the edgelist df as-is.
    """

    # FIXME: add deprecation warning for the parameter 'seed' and rename it
    # 'random_state'
    random_state = seed
    multi_gpu = False
    src, dst, weights, edge_id, edge_type = pylibcugraph_generate_rmat_edgelist(
        ResourceHandle(),
        random_state,
        scale,
        num_edges,
        a,
        b,
        c,
        clip_and_flip,
        scramble_vertex_ids,
        include_edge_weights,
        minimum_weight,
        maximum_weight,
        dtype,
        include_edge_ids,
        include_edge_types,
        min_edge_type_value,
        max_edge_type_value,
        multi_gpu,
    )

    df = cudf.DataFrame()
    df["src"] = src
    df["dst"] = dst

    if include_edge_weights:
        df["weights"] = weights
        weights = "weights"

    if include_edge_ids:
        df["edge_id"] = edge_id
        edge_id = "edge_id"

    if include_edge_types:
        df["edge_type"] = edge_type
        edge_type = "edge_type"

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
    G.from_cudf_edgelist(
        df,
        source="src",
        destination="dst",
        weight=weights,
        edge_id=edge_id,
        edge_type=edge_type,
        renumber=False,
    )

    return G


def convert_to_cudf(cp_arrays):
    cp_src, cp_dst, cp_edge_weights, cp_edge_ids, cp_edge_types = cp_arrays

    df = cudf.DataFrame()
    df["src"] = cp_src
    df["dst"] = cp_dst

    if cp_edge_weights is not None:
        df["weights"] = cp_edge_weights
    if cp_edge_ids is not None:
        df["edge_id"] = cp_edge_ids
    if cp_edge_types is not None:
        df["edge_type"] = cp_edge_types

    return df


def _mg_rmat(
    scale,
    num_edges,
    a,
    b,
    c,
    seed,
    clip_and_flip,
    scramble_vertex_ids,
    include_edge_weights,
    minimum_weight,
    maximum_weight,
    dtype,
    include_edge_ids,
    include_edge_types,
    min_edge_type_value,
    max_edge_type_value,
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
    result = []
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
            include_edge_weights,
            minimum_weight,
            maximum_weight,
            dtype,
            include_edge_ids,
            include_edge_types,
            min_edge_type_value,
            max_edge_type_value,
            workers=worker_list[i],
        )
        result.append(future)

    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)

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

    weights = None
    edge_id = None
    edge_type = None

    if "weights" in ddf.columns:
        weights = "weights"

    if "edge_id" in ddf.columns:
        edge_id = "edge_id"

    if "edge_type" in ddf.columns:
        edge_type = "edge_type"

    G.from_dask_cudf_edgelist(
        ddf,
        source="src",
        destination="dst",
        weight=weights,
        edge_id=edge_id,
        edge_type=edge_type,
    )

    return G


def _call_rmat(
    sID,
    scale,
    num_edges_for_worker,
    a,
    b,
    c,
    random_state,
    clip_and_flip,
    scramble_vertex_ids,
    include_edge_weights,
    minimum_weight,
    maximum_weight,
    dtype,
    include_edge_ids,
    include_edge_types,
    min_edge_type_value,
    max_edge_type_value,
):
    """
    Callable passed to dask client.submit calls that extracts the individual
    worker handle based on the dask session ID
    """
    multi_gpu = True

    return pylibcugraph_generate_rmat_edgelist(
        ResourceHandle(Comms.get_handle(sID).getHandle()),
        random_state,
        scale,
        num_edges_for_worker,
        a,
        b,
        c,
        clip_and_flip,
        scramble_vertex_ids,
        include_edge_weights,
        minimum_weight,
        maximum_weight,
        dtype,
        include_edge_ids,
        include_edge_types,
        min_edge_type_value,
        max_edge_type_value,
        multi_gpu,
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
    a=0.57,
    b=0.19,
    c=0.19,
    seed=42,
    clip_and_flip=False,
    scramble_vertex_ids=False,
    include_edge_weights=False,
    minimum_weight=None,
    maximum_weight=None,
    dtype=None,
    include_edge_ids=False,
    include_edge_types=False,
    min_edge_type_value=None,
    max_edge_type_value=None,
    create_using=cugraph.Graph,
    mg=False,
):
    """
    Generate a Graph object using a Recursive MATrix (R-MAT) graph generation
    algorithm.

    Parameters
    ----------
    scale : int
        Scale factor to set the number of vertices in the graph. Vertex IDs have
        values in [0, V), where V = 1 << 'scale'.

    num_edges : int
        Number of edges to generate

    a : float, optional (default=0.57)
        Probability of the edge being in the first partition
        The Graph 500 spec sets this value to 0.57.

    b : float, optional (default=0.19)
        Probability of the edge being in the second partition
        The Graph 500 spec sets this value to 0.19.


    c : float, optional (default=0.19)
        Probability of the edge being in the third partition
        The Graph 500 spec sets this value to 0.19.

    seed : int, optional (default=42)
        Seed value for the random number generator.

    clip_and_flip : bool, optional (default=False)
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix
        (if set to True) or not (if set to 'false).

    scramble_vertex_ids : bool, optional (default=False)
        Flag controlling whether to scramble vertex ID bits (if set to `true`)
        or not (if set to `false`); scrambling vertex ID bits breaks
        correlation between vertex ID values and vertex degrees.

    include_edge_weights : bool, optional (default=False)
        Flag controlling whether to generate edges with weights
        (if set to True) or not (if set to False).

    minimum_weight : float
        Minimum weight value to generate if 'include_edge_weights' is True
        otherwise, this parameter is ignored.

    maximum_weight : float
        Maximum weight value to generate if 'include_edge_weights' is True
        otherwise, this parameter is ignored.

    dtype : numpy.float32, numpy.float64, cupy.float32, cupy.float64,
    "float32", "float64"
        The type of weight to generate which is ignored unless
        include_weights is true.

    include_edge_ids : bool, optional (default=False)
        Flag controlling whether to generate edges with ids
        (if set to True) or not (if set to False).

    include_edge_types : bool, optional (default=False)
        Flag controlling whether to generate edges with types
        (if set to True) or not (if set to False).

    min_edge_type_value : int
        Minimum edge type to generate if 'include_edge_types' is True
        otherwise, this parameter is ignored.

    max_edge_type_value : int
        Maximum edge type to generate if 'include_edge_types' is True
        otherwise, this paramter is ignored.

    create_using : cugraph Graph type or None The graph type to construct
        containing the generated edges and vertices.  If None is specified, the
        edgelist cuDF DataFrame (or dask_cudf DataFrame for MG) is returned
        as-is. This is useful for benchmarking Graph construction steps that
        require raw data that includes potential self-loops, isolated vertices,
        and duplicated edges.  Default is cugraph.Graph.

    mg : bool, optional (default=False)
        If True, R-MAT generation occurs across multiple GPUs. If False, only a
        single GPU is used.  Default is False (single-GPU).

    Returns
    -------
    instance of cugraph.Graph or cudf or dask_cudf DataFrame

    Examples
    --------
    >>> import cugraph
    >>> from cugraph.generators import rmat
    >>> scale = 10
    >>> edgefactor = 16
    >>> df = rmat(
    ...    scale,
    ...    (2**scale)*edgefactor,
    ...    0.57,
    ...    0.19,
    ...    0.19,
    ...    seed=42,
    ...    clip_and_flip=False,
    ...    scramble_vertex_ids=True,
    ...    create_using=None,
    ...    mg=False
    ... )

    """

    _ensure_args_rmat(
        scale=scale,
        num_edges=num_edges,
        a=a,
        b=b,
        c=c,
        seed=seed,
        clip_and_flip=clip_and_flip,
        scramble_vertex_ids=scramble_vertex_ids,
        include_edge_weights=include_edge_weights,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        dtype=dtype,
        include_edge_ids=include_edge_ids,
        include_edge_types=include_edge_types,
        min_edge_type_value=min_edge_type_value,
        max_edge_type_value=max_edge_type_value,
        create_using=create_using,
        mg=mg,
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
            include_edge_weights,
            minimum_weight,
            maximum_weight,
            dtype,
            include_edge_ids,
            include_edge_types,
            min_edge_type_value,
            max_edge_type_value,
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
            include_edge_weights,
            minimum_weight,
            maximum_weight,
            dtype,
            include_edge_ids,
            include_edge_types,
            min_edge_type_value,
            max_edge_type_value,
            create_using,
        )


def multi_rmat(
    n_edgelists,
    min_scale,
    max_scale,
    edge_factor,
    size_distribution,
    edge_distribution,
    seed=42,
    clip_and_flip=False,
    scramble_vertex_ids=False,
    include_edge_weights=False,
    minimum_weight=None,
    maximum_weight=None,
    dtype=None,
    include_edge_ids=False,
    include_edge_types=False,
    min_edge_type_value=None,
    max_edge_type_value=None,
    mg=False,
):
    """
    Generate multiple Graph objects using a Recursive MATrix (R-MAT) graph
    generation algorithm.

    Parameters
    ----------
    n_edgelists : int
        Number of edge lists (graphs) to generate.

    min_scale : int
        Scale factor to set the minimum number of vertices in the graph.

    max_scale : int
        Scale factor to set the maximum number of vertices in the graph.

    edge_factor : int
        Average number of edges per vertex to generate

    size_distribution : int
        Distribution of the graph sizes, impacts the scale parameter of the
        R-MAT generator.
        '0' for POWER_LAW distribution and '1' for UNIFORM distribution.

    edge_distribution : int
        Edges distribution for each graph, impacts how R-MAT parameters
        a,b,c,d, are set.
        '0' for POWER_LAW distribution and '1' for UNIFORM distribution.

    seed : int
        Seed value for the random number generator.

    clip_and_flip : bool, optional (default=False)
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix
        (if set to True) or not (if set to False).

    scramble_vertex_ids : bool
        Flag controlling whether to scramble vertex ID bits (if set to True)
        or not (if set to False); scrambling vertx ID bits breaks correlation
        between vertex ID values and vertex degrees.

    include_edge_weights : bool, optional (default=False)
        Flag controlling whether to generate edges with weights
        (if set to True) or not (if set to ').

    minimum_weight : float
        Minimum weight value to generate if 'include_edge_weights' is True
        otherwise, this parameter is ignored.

    maximum_weight : float
        Maximum weight value to generate if 'include_edge_weights' is True
        otherwise, this parameter is ignored.

    include_edge_ids : bool, optional (default=False)
        Flag controlling whether to generate edges with ids
        (if set to True) or not (if set to False).

    include_edge_types : bool, optional (default=False)
        Flag controlling whether to generate edges with types
        (if set to True) or not (if set to False).

    min_edge_type_value : int
        Minimum edge type to generate if 'include_edge_types' is True
        otherwise, this parameter is ignored.

    max_edge_type_value : int
        Maximum edge type to generate if 'include_edge_types' is True
        otherwise, this paramter is ignored.

    dtype : numpy.float32, numpy.float64, cupy.float32, cupy.float64,
    "float32", "float64"
        The type of weight to generate which is ignored unless
        include_weights is true.

    mg : bool, optional (default=False)
        If True, R-MATs generation occurs across multiple GPUs. If False, only a
        single GPU is used.  Default is False (single-GPU)
        # FIXME: multi GPU RMATs generation not supported yet.

    Returns
    -------
    list of cugraph.Graph instances
    """
    _ensure_args_rmat(
        n_edgelists=n_edgelists,
        min_scale=min_scale,
        max_scale=max_scale,
        edge_factor=edge_factor,
        size_distribution=size_distribution,
        edge_distribution=edge_distribution,
        seed=seed,
        include_edge_weights=include_edge_weights,
        minimum_weight=minimum_weight,
        maximum_weight=maximum_weight,
        dtype=dtype,
        include_edge_ids=include_edge_ids,
        include_edge_types=include_edge_types,
        min_edge_type_value=min_edge_type_value,
        max_edge_type_value=max_edge_type_value,
        multi_rmat=True,
        clip_and_flip=clip_and_flip,
        scramble_vertex_ids=scramble_vertex_ids,
    )

    edgelists = pylibcugraph_generate_rmat_edgelists(
        ResourceHandle(),
        seed,
        n_edgelists,
        min_scale,
        max_scale,
        edge_factor,
        size_distribution,
        edge_distribution,
        clip_and_flip,
        scramble_vertex_ids,
        include_edge_weights,
        minimum_weight,
        maximum_weight,
        dtype,
        include_edge_ids,
        include_edge_types,
        min_edge_type_value,
        max_edge_type_value,
        mg,
    )

    dfs = []

    for edgelist in edgelists:
        src, dst, weights, edge_id, edge_type = edgelist
        df = cudf.DataFrame()
        df["src"] = src
        df["dst"] = dst
        if weights is not None:
            df["weights"] = weights
            weights = "weights"

        if edge_id is not None:
            df["edge_id"] = edge_id
            edge_id = "edge_id"
        if edge_type is not None:
            df["edge_type"] = edge_type
            edge_type = "edge_type"

        dfs.append(df)

    list_G = []

    for df in dfs:
        G = cugraph.Graph()
        G.from_cudf_edgelist(
            df,
            source="src",
            destination="dst",
            weight=weights,
            edge_id=edge_id,
            edge_type=edge_type,
        )
        list_G.append(G)

    return list_G
