# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
from cugraph.dask.common.input_utils import get_distributed_data
from cugraph.structure.shuffle import shuffle
from cugraph.dask.traversal import mg_bfs_wrapper as mg_bfs
import cugraph.comms.comms as Comms
import cudf
import dask_cudf


def call_bfs(sID,
             data,
             num_verts,
             num_edges,
             vertex_partition_offsets,
             start,
             return_distances):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_bfs.mg_bfs(data[0],
                         num_verts,
                         num_edges,
                         vertex_partition_offsets,
                         wid,
                         handle,
                         start,
                         return_distances)


def bfs(graph,
        start,
        return_distances=False):
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
    return_distances : bool, optional, default=False
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
    >>> Comms.initialize(p2p=True)
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.DiGraph()
    >>> dg.from_dask_cudf_edgelist(ddf, 'src', 'dst')
    >>> df = dcg.bfs(dg, 0)
    >>> Comms.destroy()
    """

    client = default_client()

    graph.compute_renumber_edge_list(transposed=False)
    (ddf,
     num_verts,
     partition_row_size,
     partition_col_size,
     vertex_partition_offsets) = shuffle(graph, transposed=False)
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    if graph.renumbered:
        start = graph.lookup_internal_vertex_id(cudf.Series([start],
                                                dtype='int32')).compute()
        start = start.iloc[0]

    result = [client.submit(
              call_bfs,
              Comms.get_session_id(),
              wf[1],
              num_verts,
              num_edges,
              vertex_partition_offsets,
              start,
              return_distances,
              workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)

    if graph.renumbered:
        ddf = graph.unrenumber(ddf, 'vertex')
        ddf = graph.unrenumber(ddf, 'predecessor')
        ddf["predecessor"] = ddf["predecessor"].fillna(-1)
    return ddf
