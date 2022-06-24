# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from pylibcugraph import (MGGraph,
                          ResourceHandle,
                          GraphProperties,
                          bfs as pylibcugraph_bfs
                          )

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data
import cugraph.dask.comms.comms as Comms
import cudf
import dask_cudf


def _call_plc_mg_bfs(
                    sID,
                    data,
                    sources,
                    depth_limit,
                    src_col_name,
                    dst_col_name,
                    graph_properties,
                    num_edges,
                    direction_optimizing=False,
                    do_expensive_check=False,
                    return_predecessors=True):
    comms_handle = Comms.get_handle(sID)
    resource_handle = ResourceHandle(comms_handle.getHandle())
    sources = sources[0]
    srcs = cudf.Series(data[0][src_col_name], dtype='int32')
    dsts = cudf.Series(data[0][dst_col_name], dtype='int32')
    weights = cudf.Series(data[0]['value'], dtype='float32') \
        if 'value' in data[0].columns \
        else cudf.Series((srcs + 1) / (srcs + 1), dtype='float32')

    mg = MGGraph(
        resource_handle=resource_handle,
        graph_properties=graph_properties,
        src_array=srcs,
        dst_array=dsts,
        weight_array=weights,
        store_transposed=False,
        num_edges=num_edges,
        do_expensive_check=do_expensive_check
    )

    res = \
        pylibcugraph_bfs(
            resource_handle,
            mg,
            cudf.Series(sources, dtype='int32'),
            direction_optimizing,
            depth_limit if depth_limit is not None else 0,
            return_predecessors,
            True
        )

    return res


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


def bfs(input_graph,
        start,
        depth_limit=None,
        return_distances=True,
        check_start=True):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.

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

    input_graph.compute_renumber_edge_list(
        transposed=False, legacy_renum_only=True)
    ddf = input_graph.edgelist.edgelist_df

    graph_properties = GraphProperties(
        is_multigraph=False)

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    if not isinstance(start, (dask_cudf.DataFrame, dask_cudf.Series)):
        if not isinstance(start, (cudf.DataFrame, cudf.Series)):
            start = cudf.Series(start)
        if isinstance(start, (cudf.DataFrame, cudf.Series)):
            # convert into a dask_cudf
            start = dask_cudf.from_cudf(start, ddf.npartitions)

    def check_valid_vertex(G, start):
        is_valid_vertex = G.has_node(start)
        if not is_valid_vertex:
            raise ValueError(
                'At least one start vertex provided was invalid')

    if check_start:
        check_valid_vertex(input_graph, start)

    if input_graph.renumbered:
        if isinstance(start, dask_cudf.DataFrame):
            tmp_col_names = start.columns

        elif isinstance(start, dask_cudf.Series):
            tmp_col_names = None

        start = input_graph.lookup_internal_vertex_id(
            start, tmp_col_names)

    data_start = get_distributed_data(start)

    cupy_result = [client.submit(
              _call_plc_mg_bfs,
              Comms.get_session_id(),
              wf[1],
              wf_start[1],
              depth_limit,
              src_col_name,
              dst_col_name,
              graph_properties,
              num_edges,
              False,
              True,
              return_distances,
              workers=[wf[0]])
              for idx, (wf, wf_start) in enumerate(
                  zip(data.worker_to_parts.items(),
                      data_start.worker_to_parts.items()))]

    wait(cupy_result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays)
                   for cp_arrays in cupy_result]
    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, 'vertex')
        ddf = input_graph.unrenumber(ddf, 'predecessor')
        ddf = ddf.fillna(-1)
    return ddf
