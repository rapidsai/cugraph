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
from cugraph.dask.traversal import mg_sssp_wrapper as mg_sssp
import cugraph.comms.comms as Comms
import cudf
import dask_cudf


def call_sssp(sID,
              data,
              num_verts,
              num_edges,
              vertex_partition_offsets,
              aggregate_segment_offsets,
              start):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    local_size = len(aggregate_segment_offsets) // Comms.get_n_workers(sID)
    segment_offsets = \
        aggregate_segment_offsets[local_size * wid: local_size * (wid + 1)]
    return mg_sssp.mg_sssp(data[0],
                           num_verts,
                           num_edges,
                           vertex_partition_offsets,
                           wid,
                           handle,
                           segment_offsets,
                           start)


def sssp(graph,
         source):

    """
    Compute the distance and predecessors for shortest paths from the specified
    source to all the vertices in the graph. The distances column will store
    the distance from the source to each vertex. The predecessors column will
    store each vertex's predecessor in the shortest path. Vertices that are
    unreachable will have a distance of infinity denoted by the maximum value
    of the data type and the predecessor set as -1. The source vertex's
    predecessor is also set to -1.
    The input graph must contain edge list as dask-cudf dataframe with
    one partition per GPU.

    Parameters
    ----------
    graph : cugraph.DiGraph
        cuGraph graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe.
        Undirected Graph not currently supported.
    source : Integer
        Specify source vertex

    Returns
    -------
    df : dask_cudf.DataFrame
        df['vertex'] gives the vertex id

        df['distance'] gives the path distance from the
        starting vertex

        df['predecessor'] gives the vertex id it was
        reached from in the traversal

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> ... Init a DASK Cluster
    >>    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.DiGraph()
    >>> dg.from_dask_cudf_edgelist(ddf, 'src', 'dst')
    >>> df = dcg.sssp(dg, 0)
    """

    client = default_client()

    graph.compute_renumber_edge_list(transposed=False)
    ddf = graph.edgelist.edgelist_df
    vertex_partition_offsets = get_vertex_partition_offsets(graph)
    num_verts = vertex_partition_offsets.iloc[-1]
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    if graph.renumbered:
        source = graph.lookup_internal_vertex_id(cudf.Series([source])
                                                 ).compute()
        source = source.iloc[0]

    result = [client.submit(
              call_sssp,
              Comms.get_session_id(),
              wf[1],
              num_verts,
              num_edges,
              vertex_partition_offsets,
              graph.aggregate_segment_offsets,
              source,
              workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)

    if graph.renumbered:
        ddf = graph.unrenumber(ddf, 'vertex')
        ddf = graph.unrenumber(ddf, 'predecessor')
        ddf["predecessor"] = ddf["predecessor"].fillna(-1)

    return ddf
