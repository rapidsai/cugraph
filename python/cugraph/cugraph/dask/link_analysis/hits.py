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

import cugraph.comms.comms as Comms
import dask_cudf
from dask_cudf.core import DataFrame as dcDataFrame

from pylibcugraph.experimental import (ResourceHandle,
                                       GraphProperties,
                                       MGGraph,
                                       hits,
                                       )


def call_MGGraph():
    srcs = part.columns[0]
    dsts = part.columns[1]
    weights = part.columns[2]
    # FIXME Only float 32 supported. is it still true ?
    weights = weights.astype("float32")

def hits(data, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
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
    # FIXME: Should one of the input be a dask_cudf or a cugraph.Graph
    data : cugraph.Graph or dask_cudf
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        The adjacency list will be computed if not already present.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.
        The gunrock implementation does not currently support tolerance,
        so this will in fact be the number of iterations the HITS algorithm
        executes.

    tol : float, optional (default=1.0e-5)
        Set the tolerance the approximation, this parameter should be a small
        magnitude value.  This parameter is not currently supported.

    nstart : cudf.Dataframe, optional (default=None)
        Not currently supported

    normalized : bool, optional (default=True)
        Not currently supported, always used as True

    Returns
    -------
    # FIXME Maybe a dask_cudf to be compatible with the tests?

    A tuple of device arrays, where the third item in the tuple is a device
    array containing the vertex identifiers, the first and second items are device
    arrays containing respectively the hubs and authorities values for the corresponding
    vertices
    """
    resource_handle = ResourceHandle()
    graph_properties = GraphProperties(is_symmetric=False, is_multigraph=False)
    input_type = type(data)

    # FIXME The 'data' can be represented as a cugraph.Graph or dask_cudf
    # Should cudf or other input type(cupy, ...) be also supported? if yes probably do that outside the algo call?
    if isinstance(data, Graph):
        # If the user alreday renumbered the data with Graph.compute_renumber_edge_list(), still take the unrenumbered
        # df and which will be renumbered when creating the pylibcugraph mg_graph 
        edgelist_df = data.input_df
    
    # FIXME This assume only 3 columns in the following order source, destination and edge attribute
    elif isinstance(data, dcDataFrame):
        edgelist_df = data
    
    # FIXME Only cugraph.Graph and dask_cudf are supported now
    else:
        raise TypeError("input must be either a cuGraph or dask_cudf graph "
                        f"type, got {input_type}")

    # Extract srcs, dsts and weights
    # srcs = edgelist_df["src"]
    # dsts = edgelist_df["dst"]
    # weights = edgelist_df["weight"]
    # weights = weights.astype("float32")

    

    # FIXME: Make sure this is freed by the gc.collect
    data = []

    worker_list = Comms.get_workers()

    persisted = [client.persist(
                edgelist_df.get_partition(p), workers=w) for p, w in enumerate(
                    worker_list)]
    parts = futures_of(persisted)
    wait(parts)

    G = [client.submit(call_MGGraph, part) for part in parts]


    #G = call_MGGraph()



