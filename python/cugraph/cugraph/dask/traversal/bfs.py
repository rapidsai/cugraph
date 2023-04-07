# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#

from pylibcugraph import ResourceHandle, bfs as pylibcugraph_bfs

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data
import cugraph.dask.comms.comms as Comms
import cudf
import dask_cudf
import warnings


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_distances, cupy_predecessors, cupy_vertices = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["distance"] = cupy_distances
    df["predecessor"] = cupy_predecessors
    return df


def _call_plc_bfs(
    sID,
    mg_graph_x,
    st_x,
    depth_limit=None,
    direction_optimizing=False,
    return_distances=True,
    do_expensive_check=False,
):
    return pylibcugraph_bfs(
        ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        sources=st_x,
        direction_optimizing=direction_optimizing,
        depth_limit=depth_limit if depth_limit is not None else 0,
        compute_predecessors=return_distances,
        do_expensive_check=do_expensive_check,
    )


def bfs(input_graph, start, depth_limit=None, return_distances=True, check_start=True):
    """
    Find the distances and predecessors for a breadth-first traversal of a
    graph.
    The input graph must contain edge list as a dask-cudf dataframe with
    one partition per GPU.

    Note: This is a pylibcugraph-enabled algorithm, which requires that the
    graph was created with legacy_renum_only=True.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph instance, should contain the connectivity information
        as dask cudf edge list dataframe (edge weights are not used for this
        algorithm).

    start : Integer or list or cudf object or dask_cudf object
        The id(s) of the graph vertex from which the traversal begins
        in each component of the graph.  Only one vertex per connected
        component of the graph is allowed.

    depth_limit : Integer or None, optional (default=None)
        Limit the depth of the search

    return_distances : bool, optional (default=True)
        Indicates if distances should be returned

    check_start : bool, optional (default=True)
        If True, performs more extensive tests on the start vertices
        to ensure validitity, at the expense of increased run time.

    Returns
    -------
    df : dask_cudf.DataFrame
        df['vertex'] gives the vertex id

        df['distance'] gives the path distance from the
        starting vertex (Only if return_distances is True)

        df['predecessor'] gives the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> import dask_cudf
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> ddf = dask_cudf.read_csv(datasets_path / "karate.csv",
    ...                          chunksize=chunksize, delimiter=" ",
    ...                          names=["src", "dst", "value"],
    ...                          dtype=["int32", "int32", "float32"])
    >>> dg = cugraph.Graph(directed=True)
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    ...                            edge_attr='value')
    >>> df = dcg.bfs(dg, 0)

    """

    client = default_client()
    invalid_dtype = False

    if not isinstance(start, (dask_cudf.DataFrame, dask_cudf.Series)):
        if not isinstance(start, (cudf.DataFrame, cudf.Series)):
            vertex_dtype = input_graph.nodes().dtype
            start = cudf.Series(start, dtype=vertex_dtype)
        # convert into a dask_cudf
        start = dask_cudf.from_cudf(start, input_graph._npartitions)

    if check_start:
        if isinstance(start, dask_cudf.Series):
            vertex_dtype = input_graph.nodes().dtype
            if start.dtype is not vertex_dtype:
                invalid_dtype = True
        else:
            # Multicolumn vertices case
            start_dtype = start.dtypes.reset_index(drop=True)
            vertex_dtype = input_graph.nodes().dtypes.reset_index(drop=True)
            if not start_dtype.equals(vertex_dtype):
                invalid_dtype = True

        if invalid_dtype:
            warning_msg = (
                "The 'start' values dtype must match " "the graph's vertices dtype."
            )

            warnings.warn(warning_msg, UserWarning)
            if isinstance(start, dask_cudf.Series):
                start = start.astype(vertex_dtype)
            else:
                start = start.astype(vertex_dtype[0])

        is_valid_vertex = input_graph.has_node(start)
        if not is_valid_vertex:
            raise ValueError("At least one start vertex provided was invalid")

    if input_graph.renumbered:
        if isinstance(start, dask_cudf.DataFrame):
            tmp_col_names = start.columns

        elif isinstance(start, dask_cudf.Series):
            tmp_col_names = None

        start = input_graph.lookup_internal_vertex_id(start, tmp_col_names)

    data_start = get_distributed_data(start)
    do_expensive_check = False
    # FIXME: Why is 'direction_optimizing' not part of the python cugraph API
    # and why is it set to 'False' by default
    direction_optimizing = False

    cupy_result = [
        client.submit(
            _call_plc_bfs,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            st[0],
            depth_limit,
            direction_optimizing,
            return_distances,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w, st in data_start.worker_to_parts.items()
    ]

    wait(cupy_result)

    cudf_result = [
        client.submit(convert_to_cudf, cp_arrays) for cp_arrays in cupy_result
    ]
    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(cupy_result, cudf_result)])

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "vertex")
        ddf = input_graph.unrenumber(ddf, "predecessor")
        ddf = ddf.fillna(-1)
    return ddf
