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
from cugraph.dask.common.input_utils import get_local_data
from cugraph.dask.traversal import mg_bfs_wrapper as mg_bfs
import cugraph.comms.comms as Comms
import cudf


def call_bfs(sID, data, local_data, start, num_verts, return_distances):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_bfs.mg_bfs(data[0],
                         local_data,
                         wid,
                         handle,
                         start,
                         num_verts,
                         return_distances)


def bfs(graph,
        start,
        return_distances=False,
        load_balance=True):

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
    load_balance : bool, optional, default=True
        Set as True to perform load_balancing after global sorting of
        dask-cudf DataFrame. This ensures that the data is uniformly
        distributed among multiple GPUs to avoid over-loading.

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex

        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex (Only if return_distances is True)

        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> Comms.initialize()
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.DiGraph()
    >>> dg.from_dask_cudf_edgelist(ddf)
    >>> df = dcg.bfs(dg, 0)
    >>> Comms.destroy()
    """

    client = default_client()

    if(graph.local_data is not None and
       graph.local_data['by'] == 'src'):
        data = graph.local_data['data']
    else:
        data = get_local_data(graph, by='src', load_balance=load_balance)

    if graph.renumbered:
        start = graph.lookup_internal_vertex_id(cudf.Series([start],
                                                dtype='int32')).compute()
        start = start.iloc[0]

    print('start = ', start)

    result = dict([(data.worker_info[wf[0]]["rank"],
                    client.submit(
            call_bfs,
            Comms.get_session_id(),
            wf[1],
            data.local_data,
            start,
            data.max_vertex_id+1,
            return_distances,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])
    wait(result)

    df = result[0].result()

    print('df = \n', df)

    if graph.renumbered:
        df = graph.unrenumber(df, 'vertex').compute()
        df = graph.unrenumber(df, 'predecessor').compute()
        df["predecessor"].fillna(-1, inplace=True)

    return df
