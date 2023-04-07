# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

import cugraph.dask.comms.comms as Comms
import dask_cudf
import cudf
from cugraph.dask.common.input_utils import get_distributed_data

from pylibcugraph import ResourceHandle, ego_graph as pylibcugraph_ego_graph


def _call_ego_graph(
    sID,
    mg_graph_x,
    n,
    radius,
    do_expensive_check,
):
    return pylibcugraph_ego_graph(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        source_vertices=n,
        radius=radius,
        do_expensive_check=do_expensive_check,
    )


def consolidate_results(df, offsets):
    """
    Each rank returns its ego_graph dataframe with its corresponding
    offsets array. This is ideal if the user operates on distributed memory
    but when attempting to bring the result into a single machine,
    the ego_graph dataframes generated from each seed cannot be extracted
    using the offsets array. This function consolidate the final result by
    performing segmented copies.

    Returns: consolidated ego_graph dataframe
    """
    for i in range(len(offsets) - 1):
        df_tmp = df[offsets[i] : offsets[i + 1]]
        df_tmp["labels"] = i
        if i == 0:
            df_consolidate = df_tmp
        else:
            df_consolidate = cudf.concat([df_consolidate, df_tmp])
    return df_consolidate


def convert_to_cudf(cp_arrays):
    cp_src, cp_dst, cp_weight, cp_offsets = cp_arrays

    df = cudf.DataFrame()
    df["src"] = cp_src
    df["dst"] = cp_dst
    df["weight"] = cp_weight

    offsets = cudf.Series(cp_offsets)

    return consolidate_results(df, offsets)


def ego_graph(input_graph, n, radius=1, center=True):
    """
    Compute the induced subgraph of neighbors centered at node n,
    within a given radius.

    Parameters
    ----------
    input_graph : cugraph.Graph, networkx.Graph
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    n : int, list or cudf Series or Dataframe, dask_cudf Series or DataFrame
        A node or a list or cudf.Series of nodes or a cudf.DataFrame if nodes
        are represented with multiple columns. If a cudf.DataFrame is provided,
        only the first row is taken as the node input.

    radius: integer, optional (default=1)
        Include all neighbors of distance<=radius from n.

    center: bool, optional
        Defaults to True. False is not supported

    Returns
    -------
    ego_edge_lists : dask_cudf.DataFrame
        Distributed GPU data frame containing all induced sources identifiers,
        destination identifiers, edge weights
    seeds_offsets: dask_cudf.Series
        Distributed Series containing the starting offset in the returned edge list
        for each seed.

    """

    # Initialize dask client
    client = default_client()

    if isinstance(n, (int, list)):
        n = cudf.Series(n)
    elif not isinstance(
        n, (cudf.Series, dask_cudf.Series, cudf.DataFrame, dask_cudf.DataFrame)
    ):
        raise TypeError(
            f"'n' must be either an integer or a list or a "
            f"cudf or dask_cudf Series or DataFrame, got: {type(n)}"
        )

    # n uses "external" vertex IDs, but since the graph has been
    # renumbered, the node ID must also be renumbered.
    if input_graph.renumbered:
        n = input_graph.lookup_internal_vertex_id(n)
        n_type = input_graph.edgelist.edgelist_df.dtypes[0]
    else:
        n_type = input_graph.input_df.dtypes[0]

    if isinstance(n, (cudf.Series, cudf.DataFrame)):
        n = dask_cudf.from_cudf(n, npartitions=min(input_graph._npartitions, len(n)))
    n = n.astype(n_type)

    n = get_distributed_data(n)
    wait(n)

    n = n.worker_to_parts

    do_expensive_check = False

    result = [
        client.submit(
            _call_ego_graph,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            n[w][0],
            radius,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w in Comms.get_workers()
    ]
    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    ddf = ddf.sort_values("labels")

    # extract offsets from segmented ego_graph dataframes
    offsets = ddf["labels"].value_counts().compute().sort_index()
    offsets = cudf.concat([cudf.Series(0), offsets])
    offsets = (
        dask_cudf.from_cudf(offsets, npartitions=min(input_graph._npartitions, len(n)))
        .cumsum()
        .astype(n_type)
    )

    ddf = ddf.drop(columns="labels")

    return ddf, offsets
