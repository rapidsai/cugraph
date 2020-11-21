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
from cugraph.dask.centrality import\
    mg_katz_centrality_wrapper as mg_katz_centrality
import cugraph.comms.comms as Comms
import dask_cudf


def call_katz_centrality(sID,
                         data,
                         num_verts,
                         num_edges,
                         vertex_partition_offsets,
                         alpha,
                         beta,
                         max_iter,
                         tol,
                         nstart,
                         normalized):
    wid = Comms.get_worker_id(sID)
    handle = Comms.get_handle(sID)
    return mg_katz_centrality.mg_katz_centrality(data[0],
                                                 num_verts,
                                                 num_edges,
                                                 vertex_partition_offsets,
                                                 wid,
                                                 handle,
                                                 alpha,
                                                 beta,
                                                 max_iter,
                                                 tol,
                                                 nstart,
                                                 normalized)


def katz_centrality(input_graph,
                    alpha=None,
                    beta=None,
                    max_iter=100,
                    tol=1.0e-5,
                    nstart=None,
                    normalized=True):
    """
    Compute the Katz centrality for the nodes of the graph G.

    Parameters
    ----------
    input_graph : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. The graph can
        contain either directed (DiGraph) or undirected edges (Graph).
    alpha : float
        Attenuation factor defaulted to None. If alpha is not specified then
        it is internally calculated as 1/(degree_max) where degree_max is the
        maximum out degree.
        NOTE : The maximum acceptable value of alpha for convergence
        alpha_max = 1/(lambda_max) where lambda_max is the largest eigenvalue
        of the graph.
        Since lambda_max is always lesser than or equal to degree_max for a
        graph, alpha_max will always be greater than or equal to
        (1/degree_max). Therefore, setting alpha to (1/degree_max) will
        guarantee that it will never exceed alpha_max thus in turn fulfilling
        the requirement for convergence.
    beta : None
        A weight scalar - currently Not Supported
    max_iter : int
        The maximum number of iterations before an answer is returned. This can
        be used to limit the execution time and do an early exit before the
        solver reaches the convergence tolerance.
        If this value is lower or equal to 0 cuGraph will use the default
        value, which is 100.
    tolerance : float
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.
        The lower the tolerance the better the approximation. If this value is
        0.0f, cuGraph will use the default value which is 1.0e-6.
        Setting too small a tolerance can lead to non-convergence due to
        numerical roundoff. Usually values between 1e-2 and 1e-6 are
        acceptable.
    nstart : dask_cudf.Dataframe
        GPU Dataframe containing the initial guess for katz centrality
        nstart['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        nstart['values'] : dask_cudf.Series
            Contains the katz centrality values of vertices
    normalized : bool
        If True normalize the resulting katz centrality values

    Returns
    -------
    katz_centrality : dask_cudf.DataFrame
        GPU data frame containing two dask_cudf.Series of size V: the
        vertex identifiers and the corresponding katz centrality values.

        ddf['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        ddf['katz_centrality'] : dask_cudf.Series
            Contains the katz centrality of vertices

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
    >>> dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
                                   edge_attr='value')
    >>> pr = dcg.katz_centrality(dg)
    >>> Comms.destroy()
    """

    nstart = None

    client = default_client()

    input_graph.compute_renumber_edge_list(transposed=True)
    (ddf,
     num_verts,
     partition_row_size,
     partition_col_size,
     vertex_partition_offsets) = shuffle(input_graph, transposed=True)
    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    result = [client.submit(call_katz_centrality,
                            Comms.get_session_id(),
                            wf[1],
                            num_verts,
                            num_edges,
                            vertex_partition_offsets,
                            alpha,
                            beta,
                            max_iter,
                            tol,
                            nstart,
                            normalized,
                            workers=[wf[0]])
              for idx, wf in enumerate(data.worker_to_parts.items())]
    wait(result)
    ddf = dask_cudf.from_delayed(result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
