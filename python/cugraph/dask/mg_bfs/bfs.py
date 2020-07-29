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
from cugraph.mg.traversal import mg_bfs_wrapper as mg_bfs
import cugraph.comms.comms as Comms


def call_bfs(sID, data, local_data, start, return_distances):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_bfs.mg_bfs(data[0],
                         local_data,
                         wid,
                         handle,
                         start,
                         return_distances)


def bfs(input_graph,
        start,
        return_distances=False):

    """
    Find the PageRank values for each vertex in a graph using multiple GPUs.
    cuGraph computes an approximation of the Pagerank using the power method.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.

    Parameters
    ----------
    edge_list : dask_cudf.DataFrame
        Contain the connectivity information as an edge list.
        Source 'src' and destination 'dst' columns must be of type 'int32'.
        Edge weights are not used for this algorithm.
        Indices must be in the range [0, V-1], where V is the global number
        of vertices.
    start : Integer
        The index of the graph vertex from which the traversal begins

    return_sp_counter : bool, optional, default=False
        Indicates if shortest path counters should be returned

    Returns
    -------
    df : cudf.DataFrame
        df['vertex'][i] gives the vertex id of the i'th vertex

        df['distance'][i] gives the path distance for the i'th vertex from the
        starting vertex

        df['predecessor'][i] gives for the i'th vertex the vertex it was
        reached from in the traversal

    Examples
    --------
    >>> import cugraph.dask as dcg
    >>> chunksize = dcg.get_chunksize(input_data_path)
    >>> ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                                 delimiter=' ',
                                 names=['src', 'dst', 'value'],
                                 dtype=['int32', 'int32', 'float32'])
    >>> dg = cugraph.DiGraph()
    >>> dg.from_dask_cudf_edgelist(ddf)
    >>> df = dcg.bfs(dg, 0)
    """

    client = default_client()

    if(input_graph.local_data is not None and
       input_graph.local_data['by'] == 'src'):
        data = input_graph.local_data['data']
    else:
        data = get_local_data(input_graph, by='src')

    result = dict([(data.worker_info[wf[0]]["rank"],
                    client.submit(
            call_bfs,
            Comms.get_session_id(),
            wf[1],
            data.local_data,
            start,
            return_distances,
            workers=[wf[0]]))
            for idx, wf in enumerate(data.worker_to_parts.items())])
    wait(result)

    return result[0].result()
