# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import (get_distributed_data,
                                             get_vertex_partition_offsets)
from cugraph.dask.traversal import mg_bfs_wrapper as mg_bfs
import cugraph.comms.comms as Comms
import cudf
import dask_cudf


def call_bfs(sID,
             data,
             num_verts,
             num_edges,
             vertex_partition_offsets,
             aggregate_segment_offsets,
             start,
             depth_limit,
             return_distances):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    local_size = len(aggregate_segment_offsets) // Comms.get_n_workers(sID)
    segment_offsets = \
        aggregate_segment_offsets[local_size * wid: local_size * (wid + 1)]
    return mg_bfs.mg_bfs(data[0],
                         num_verts,
                         num_edges,
                         vertex_partition_offsets,
                         wid,
                         handle,
                         segment_offsets,
                         start,
                         depth_limit,
                         return_distances)


def bfs(graph,
        start,
        depth_limit=None,
        return_distances=True):
    """
    Find the distances and predecessors for a breadth first traversal of a
    graph.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.

    Parameters
    ----------
    graph : cugraph.DiGraph
        cuGraph graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe(edge weights are not used for this
        algorithm). Undirected Graph not currently supported.
    start : Integer
        Specify starting vertex for breadth-first search; this function
        iterates over edges in the component reachable from this node.
    depth_limit : Integer or None
        Limit the depth of the search
    return_distances : bool, optional, default=True
        Indicates if distances should be returned

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
    >>> ... Init a DASK Cluster
    >>    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>  Download dataset from https://github.com/rapidsai/cugraph/datasets/...
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.DiGraph()
    >>> dg.from_dask_cudf_edgelist(ddf, 'src', 'dst')
    >>> df = dcg.bfs(dg, 0)
    """

    client = default_client()

    graph.compute_renumber_edge_list(transposed=False)
    ddf = graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(graph)
    num_verts = vertex_partition_offsets.iloc[-1]

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    def df_merge(df, tmp_df, tmp_col_names):
        x = df[0].merge(tmp_df, on=tmp_col_names, how='inner')
        return x['global_id']

    if graph.renumbered:
        renumber_ddf = graph.renumber_map.implementation.ddf
        col_names = graph.renumber_map.implementation.col_names
        if isinstance(start,
                      dask_cudf.DataFrame) or isinstance(start,
                                                         cudf.DataFrame):
            tmp_df = start
            tmp_col_names = start.columns
        else:
            tmp_df = cudf.DataFrame()
            tmp_df["0"] = cudf.Series(start)
            tmp_col_names = ["0"]
        tmp_ddf = tmp_df[tmp_col_names].rename(
            columns=dict(zip(tmp_col_names, col_names)))
        for name in col_names:
            tmp_ddf[name] = tmp_ddf[name].astype(renumber_ddf[name].dtype)
        renumber_data = get_distributed_data(renumber_ddf)
        start = [client.submit(df_merge,
                               wf[1],
                               tmp_ddf,
                               col_names,
                               workers=[wf[0]])
                 for idx, wf in enumerate(renumber_data.worker_to_parts.items()
                                          )
                 ]

    result = [client.submit(
              call_bfs,
              Comms.get_session_id(),
              wf[1],
              num_verts,
              num_edges,
              vertex_partition_offsets,
              graph.aggregate_segment_offsets,
              start[idx],
              depth_limit,
              return_distances,
              workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)

    if graph.renumbered:
        ddf = graph.unrenumber(ddf, 'vertex')
        ddf = graph.unrenumber(ddf, 'predecessor')
        ddf = ddf.fillna(-1)
    return ddf
