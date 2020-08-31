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
from cugraph.dask.link_analysis import mg_pagerank_wrapper as mg_pagerank
import cugraph.comms.comms as Comms


def call_pagerank(sID, data, local_data, alpha, max_iter,
                  tol, personalization, nstart):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_pagerank.mg_pagerank(data[0],
                                   local_data,
                                   wid,
                                   handle,
                                   alpha,
                                   max_iter,
                                   tol,
                                   personalization,
                                   nstart)


def pagerank(input_graph,
             alpha=0.85,
             personalization=None,
             max_iter=100,
             tol=1.0e-5,
             nstart=None,
             load_balance=True):

    """
    Find the PageRank values for each vertex in a graph using multiple GPUs.
    cuGraph computes an approximation of the Pagerank using the power method.
    The input graph must contain edge list as  dask-cudf dataframe with
    one partition per GPU.

    Parameters
    ----------
    graph : cugraph.DiGraph
        cuGraph graph descriptor, should contain the connectivity information
        as dask cudf edge list dataframe(edge weights are not used for this
        algorithm). Undirected Graph not currently supported.
    alpha : float
        The damping factor alpha represents the probability to follow an
        outgoing edge, standard value is 0.85.
        Thus, 1.0-alpha is the probability to “teleport” to a random vertex.
        Alpha should be greater than 0.0 and strictly lower than 1.0.
    personalization : cudf.Dataframe
        GPU Dataframe containing the personalization information.

        personalization['vertex'] : cudf.Series
            Subset of vertices of graph for personalization
        personalization['values'] : cudf.Series
            Personalization values for vertices
    max_iter : int
        The maximum number of iterations before an answer is returned.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 30.
    tolerance : float
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0E-5.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 0.01 and 0.00001 are
        acceptable.
    nstart : not supported
        initial guess for pagerank
    load_balance : bool
        Set as True to perform load_balancing after global sorting of
        dask-cudf DataFrame. This ensures that the data is uniformly
        distributed among multiple GPUs to avoid over-loading.

    Returns
    -------
    PageRank : cudf.DataFrame
        GPU data frame containing two cudf.Series of size V: the vertex
        identifiers and the corresponding PageRank values.

        df['vertex'] : cudf.Series
            Contains the vertex identifiers
        df['pagerank'] : cudf.Series
            Contains the PageRank score

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
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
                                   edge_attr='value')
    >>> pr = dcg.pagerank(dg)
    >>> Comms.destroy()
    """
    from cugraph.structure.graph import null_check

    nstart = None

    client = default_client()

    if(input_graph.local_data is not None and
       input_graph.local_data['by'] == 'dst'):
        data = input_graph.local_data['data']
    else:
        data = get_local_data(input_graph, by='dst', load_balance=load_balance)

    if personalization is not None:
        null_check(personalization["vertex"])
        null_check(personalization["values"])
        if input_graph.renumbered is True:
            personalization = input_graph.add_internal_vertex_id(
                personalization, "vertex", "vertex"
            ).compute()

    result = dict([(data.worker_info[wf[0]]["rank"],
                    client.submit(
                    call_pagerank,
                    Comms.get_session_id(),
                    wf[1],
                    data.local_data,
                    alpha,
                    max_iter,
                    tol,
                    personalization,
                    nstart,
                    workers=[wf[0]]))
                   for idx, wf in enumerate(data.worker_to_parts.items())])
    wait(result)

    if input_graph.renumbered:
        return input_graph.unrenumber(result[0].result(), 'vertex').compute()

    return result[0].result()
