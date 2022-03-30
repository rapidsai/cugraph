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
import cudf

from pylibcugraph.experimental import (ResourceHandle,
                                       GraphProperties,
                                       MGGraph,
                                       hits as pylibcugraph_hits
                                       )


def call_hits(sID,
              data,
              src_col_name,
              dst_col_name,
              graph_properties,
              store_transposed,
              num_edges,
              do_expensive_check,
              tolerance,
              max_iter,
              initial_hubs_guess_vertices,
              initial_hubs_guess_value,
              normalized):

    handle = Comms.get_handle(sID)
    h = ResourceHandle(handle.getHandle())
    srcs = data[0][src_col_name]
    dsts = data[0][dst_col_name]
    weights = None
    if "value" in data[0].columns:
        weights = data[0]['value']

    mg = MGGraph(h,
                 graph_properties,
                 srcs,
                 dsts,
                 weights,
                 store_transposed,
                 num_edges,
                 do_expensive_check)

    result = pylibcugraph_hits(h,
                               mg,
                               tolerance,
                               max_iter,
                               initial_hubs_guess_vertices,
                               initial_hubs_guess_value,
                               normalized,
                               do_expensive_check)

    return result


def convert_to_cudf(cp_arrays):
    """
    create a cudf DataFrame from cupy arrays
    """
    cupy_vertices, cupy_hubs, cupy_authorities = cp_arrays
    df = cudf.DataFrame()
    df["vertex"] = cupy_vertices
    df["hubs"] = cupy_hubs
    df["authorities"] = cupy_authorities
    return df


def hits(input_graph, tol=1.0e-5, max_iter=100,  nstart=None, normalized=True):
    """
    Compute HITS hubs and authorities values for each vertex

    The HITS algorithm computes two numbers for a node.  Authorities
    estimates the node value based on the incoming links.  Hubs estimates
    the node value based on outgoing links.

    Both cuGraph and networkx implementation use a 1-norm.

    Parameters
    ----------

    input_graph : cugraph.Graph
        cuGraph graph descriptor, should contain the connectivity information
        as an edge list (edge weights are not used for this algorithm).
        The adjacency list will be computed if not already present.

    tol : float, optional (default=1.0e-5)
        Set the tolerance of the approximation, this parameter should be a
        small magnitude value.

    max_iter : int, optional (default=100)
        The maximum number of iterations before an answer is returned.

    nstart : cudf.Dataframe, optional (default=None)
        The initial hubs guess vertices along with their initial hubs guess
        value

        nstart['vertex'] : cudf.Series
            Initial hubs guess vertices
        nstart['values'] : cudf.Series
            Initial hubs guess values

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

    # FIXME Still compute renumbering at this layer in case str
    # vertex ID are passed
    input_graph.compute_renumber_edge_list(transposed=False)
    ddf = input_graph.edgelist.edgelist_df

    graph_properties = GraphProperties(
        is_multigraph=False)

    store_transposed = False
    do_expensive_check = False
    initial_hubs_guess_vertices = None
    initial_hubs_guess_values = None

    src_col_name = input_graph.renumber_map.renumbered_src_col_name
    dst_col_name = input_graph.renumber_map.renumbered_dst_col_name

    # FIXME Move this call to the function creating a directed
    # graph from a dask dataframe because duplicated edges need
    # to be dropped
    ddf = ddf.map_partitions(
        lambda df: df.drop_duplicates(subset=[src_col_name, dst_col_name]))

    num_edges = len(ddf)
    data = get_distributed_data(ddf)

    if nstart is not None:
        initial_hubs_guess_vertices = nstart['vertex']
        initial_hubs_guess_values = nstart['values']

    cupy_result = [client.submit(call_hits,
                                 Comms.get_session_id(),
                                 wf[1],
                                 src_col_name,
                                 dst_col_name,
                                 graph_properties,
                                 store_transposed,
                                 num_edges,
                                 do_expensive_check,
                                 tol,
                                 max_iter,
                                 initial_hubs_guess_vertices,
                                 initial_hubs_guess_values,
                                 normalized,
                                 workers=[wf[0]])
                   for idx, wf in enumerate(data.worker_to_parts.items())]

    wait(cupy_result)

    cudf_result = [client.submit(convert_to_cudf,
                                 cp_arrays,
                                 workers=client.who_has(
                                     cp_arrays)[cp_arrays.key])
                   for cp_arrays in cupy_result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result)
    if input_graph.renumbered:
        return input_graph.unrenumber(ddf, 'vertex')

    return ddf
