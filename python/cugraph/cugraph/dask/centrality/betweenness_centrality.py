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
#

from dask.distributed import wait
from pylibcugraph import ResourceHandle, betweenness_centrality as pylibcugraph_betweenness
import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
import warnings


def _call_plc_betweenness_centrality(
    sID, mg_graph_x, num_vertices, vertex_list, normalized, endpoints, do_expensive_check
):

    return pylibcugraph_betweenness_centrality(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        num_vertices=num_vertices,
        vertex_list=vertex_list,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=do_expensive_check,
    )


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_vertices, cupy_values = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["betweenness_centrality"] = cupy_values
    return df


def betweenness_centrality(
    input_graph,
    num_vertices=None,
    vertex_list=None,
    normalized=True,
    endpoints=False
):
    """
    Compute the betweenness centrality for all vertices of the graph G.
    Betweenness centrality is a measure of the number of shortest paths that
    pass through a vertex.  A vertex with a high betweenness centrality score
    has more paths passing through it and is therefore believed to be more
    important.

    To improve performance. rather than doing an all-pair shortest path,
    a sample of k starting vertices can be used.

    CuGraph does not currently support the 'endpoints' and 'weight' parameters
    as seen in the corresponding networkX call.

    Parameters
    ----------
    G : cuGraph.Graph
        The graph can be either directed (Graph(directed=True)) or undirected.
        Weights in the graph are ignored, the current implementation uses
        BFS traversals. If weights are provided in the edgelist, they will be
        used.

    num_vertices: int, optional (default=None)
        Number of vertices to randomly sample.

    vertex_list: list or cudf object or dask_cudf object optional (default=None)
        specify a list of vertices to use as seeds for BFS.

    normalized : bool, optional (default=True)
        If True normalize the resulting betweenness centrality values

    endpoints : bool, optional (default=False)
        If true, include the endpoints in the shortest path counts.

    Returns
    -------
    betweenness_centrality : dask_cudf.DataFrame
        GPU distributed data frame containing two dask_cudf.Series of size V:
        the vertex identifiers and the corresponding betweenness centrality values.

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['betweenness_centrality'] : dask_cudf.Series
            Contains the betweenness centrality of vertices

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
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst')
    >>> pr = dcg.betweenness_centrality(dg)

    """
    client = input_graph._client

    if input_graph.store_transposed is True:
        warning_msg = (
            "Betweenness centrality expects the 'store_transposed' flag "
            "to be set to 'False' for optimal performance during "
            "the graph creation"
        )
        warnings.warn(warning_msg, UserWarning)

    if not isinstance(vertex_list, (dask_cudf.DataFrame, dask_cudf.Series)):
        if not isinstance(vertex_list, (cudf.DataFrame, cudf.Series)):
            if isintance(vertex_list, list):
                vertex_list_dtype = input_graph.nodes().dtype
                vertex_list = cudf.Series(vertex_list, dtype=vertex_list_dtype)
            else:
                raise TypeError(
                    f"'vertex_list' must be either a list or a cudf or "
                    f"dask_cudf object cudf.DataFrame, got: {type(vertex_list)}"
            )
        # convert into a dask_cudf
        vertex_list = dask_cudf.from_cudf(vertex_list, input_graph._npartitions)

    if input_graph.renumbered:
        if isinstance(vertex_list, dask_cudf.DataFrame):
            tmp_col_names = vertex_list.columns

        elif isinstance(vertex_list, dask_cudf.Series):
            tmp_col_names = None

        vertex_list = input_graph.lookup_internal_vertex_id(
            vertex_list, tmp_col_names)
    
    vertex_list = get_distributed_data(vertex_list)
    # FIXME: should we add this parameter as an option?
    do_expensive_check = False

    cupy_result = [
        client.submit(
            _call_plc_betweenness_centrality,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            num_vertices,
            vertex_list,
            normalized,
            endpoints,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]

    wait(cupy_result)

    cudf_result = [
        client.submit(
            convert_to_cudf, cp_arrays, workers=client.who_has(cp_arrays)[cp_arrays.key]
        )
        for cp_arrays in cupy_result
    ]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    # Wait until the inactive futures are released
    wait([(r.release(), c_r.release()) for r, c_r in zip(cupy_result, cudf_result)])

    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, "vertex")

    return ddf
