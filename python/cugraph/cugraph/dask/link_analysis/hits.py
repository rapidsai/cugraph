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

from dask.distributed import wait, default_client
from cugraph.dask.common.input_utils import get_distributed_data

import cugraph.comms.comms as Comms
import dask_cudf
from dask_cudf.core import DataFrame as dcDataFrame

from pylibcugraph.experimental import (ResourceHandle,
                                       GraphProperties,
                                       MGGraph,
                                       )


def call_MGGraph(resource_handle,
                 graph_properties,
                 data,
                 store_transposed,
                 num_edges,
                 do_expensive_check,
                 ):

    srcs = data[0]["renumbered_src"]
    dsts = data[0]["renumbered_dst"]
    
    # FIXME: Check for the existence of a weight column
    weights = data[0]["value"]

    return MGGraph(resource_handle,
                   graph_properties,
                   srcs,
                   dsts,
                   weights,
                   store_transposed,
                   num_edges,
                   do_expensive_check)

def call_hits(resource_handle,
              g,
              tolerance,
              max_iter,
              n_start,
              normalized)
    return pylibcugraph.experimental.hits(resource_handle,
                                          g,
                                          tolerance,
                                          max_iter,
                                          n_start,
                                          normalized)


def hits(input_graph, tol=1.0e-5, max_iter=100,  nstart=None, normalized=True):
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    The cuGraph implementation of HITS is a wrapper around the gunrock
    implementation of HITS.

    Note that the gunrock implementation uses a 2-norm, while networkx
    uses a 1-norm.  The raw scores will be different, but the rank ordering
    should be comparable with networkx.

    Parameters
    ----------

    input_graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        The adjacency list will be computed if not already present.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.

    nstart : cudf.Dataframe, optional (default=None)
        The intial hubs guess vertices along with their intial hubs guess
        value

        nstart['vertex'] : cudf.Series
            Intial hubs guess vertices
        nstart['values'] : cudf.Series
            intial hubs guess value

    normalized : bool, optional (default=True)
        A flag to normalize the results

    Returns
    -------
    HubsAndAuthorities : dask_cudf.DataFrame
        GPU data frame containing three cudf.Series of size V: the vertex
        identifiers and the corresponding hubs values and the corresponding
        authorities values.

        df['vertex'] : dask_cudf.Series
            Contains the vertex identifiers
        df['hubs'] : dask_cudf.Series
            Contains the hubs score
        df['authorities'] : dask_cudf.Series
            Contains the authorities score

    Examples
    --------
    >>> # import cugraph.dask as dcg
    >>> # ... Init a DASK Cluster
    >>> #    see https://docs.rapids.ai/api/cugraph/stable/dask-cugraph.html
    >>> # Download dataset from https://github.com/rapidsai/cugraph/datasets/..
    >>> # chunksize = dcg.get_chunksize(datasets_path / "karate.csv")
    >>> # ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize)
    >>> # dg = cugraph.Graph(directed=True)
    >>> # dg.from_dask_cudf_edgelist(ddf, source='src', destination='dst',
    >>> #                            edge_attr='value')
    >>> # hits = dcg.hits(dg, max_iter = 50)

    """

    client = default_client()

    #FIXME Still compute renumbering at this layer in case str vertex ID are passed
    input_graph.compute_renumber_edge_list(transposed=False)
    ddf = input_graph.edgelist.edgelist_df

    resource_handle = ResourceHandle()
    graph_properties = GraphProperties(is_multigraph=False)

    store_transposed = False
    do_expensive_check = False
    num_edges = len(ddf)

    data = get_distributed_data(ddf)

    mg = [client.submit(call_MGGraph, 
                        resource_handle, part) for part in parts]

    # FIXME: This will return a list of futures. What to do next because the results
    # are still in distributed memory?
    mg_result = [client.submit(call_MGGraph,
                               resource_handle,
                               graph_properties,
                               wf[1],
                               store_transposed,
                               num_edges,
                               do_expensive_check,
                               workers=[wf[0]])
                 for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(mg_result)
    # Bring the results back? Will the results be different (each piece of the graph)
    # or will all futures point to the same graph as a whole?
    mg = client.gather(mg_result)

    # FIXME: assumption that each future is pointing to a part of the graph
    result = [client.submit(call_hits,
                          g,
                          tolerance,
                          max_iter,
                          n_start,
                          normalized
                          workers=[client.who_has(g)])
              for g in mg]
    wait(result)
    ddf = dask_cudf.from_delayed(result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')
    
    return ddf

    



