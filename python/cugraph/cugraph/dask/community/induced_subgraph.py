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
import cupy as cp
from cugraph.dask.common.part_utils import (
    persist_dask_df_equal_parts_per_worker,
)
from typing import Union, Tuple

from pylibcugraph import (
    ResourceHandle,
    induced_subgraph as pylibcugraph_induced_subgraph,
)


def _call_induced_subgraph(
    sID: bytes,
    mg_graph_x,
    vertices: cudf.Series,
    offsets: cudf.Series,
    do_expensive_check: bool,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    return pylibcugraph_induced_subgraph(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        graph=mg_graph_x,
        subgraph_vertices=vertices,
        subgraph_offsets=offsets,
        do_expensive_check=do_expensive_check,
    )


def consolidate_results(df: cudf.DataFrame, offsets: cudf.Series) -> cudf.DataFrame:
    """
    Each rank returns its induced_subgraph dataframe with its corresponding
    offsets array. This is ideal if the user operates on distributed memory
    but when attempting to bring the result into a single machine,
    the induced_subgraph dataframes generated from each seed cannot be extracted
    using the offsets array. This function consolidate the final result by
    performing segmented copies.

    Returns: consolidated induced_subgraph dataframe
    """
    for i in range(len(offsets) - 1):
        df_tmp = df[offsets[i] : offsets[i + 1]]
        df_tmp["labels"] = i
        if i == 0:
            df_consolidate = df_tmp
        else:
            df_consolidate = cudf.concat([df_consolidate, df_tmp])
    return df_consolidate


def convert_to_cudf(cp_arrays: cp.ndarray) -> cudf.DataFrame:
    cp_src, cp_dst, cp_weight, cp_offsets = cp_arrays

    df = cudf.DataFrame()
    df["src"] = cp_src
    df["dst"] = cp_dst
    df["weight"] = cp_weight

    offsets = cudf.Series(cp_offsets)

    return consolidate_results(df, offsets)


def induced_subgraph(
    input_graph,
    vertices: Union[cudf.Series, cudf.DataFrame],
    offsets: Union[list, cudf.Series] = None,
) -> Tuple[dask_cudf.DataFrame, dask_cudf.Series]:
    """
    Compute a subgraph of the existing graph including only the specified
    vertices.  This algorithm works with both directed and undirected graphs
    and does not actually traverse the edges, but instead simply pulls out any
    edges that are incident on vertices that are both contained in the vertices
    list.

    If no subgraph can be extracted from the vertices provided, a 'None' value
    will be returned.

    Parameters
    ----------
    input_graph : cugraph.Graph
        Graph or matrix object, which should contain the connectivity
        information. Edge weights, if present, should be single or double
        precision floating point values.

    vertices : cudf.Series or cudf.DataFrame
        Specifies the vertices of the induced subgraph. For multi-column
        vertices, vertices should be provided as a cudf.DataFrame

    offsets : list or cudf.Series, optional
        Specifies the subgraph offsets into the subgraph vertices.
        If no offsets array is provided, a default array [0, len(vertices)]
        will be used.

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

    if isinstance(vertices, (int, list)):
        vertices = cudf.Series(vertices)
    elif not isinstance(
        vertices, (cudf.Series, dask_cudf.Series, cudf.DataFrame, dask_cudf.DataFrame)
    ):
        raise TypeError(
            f"'vertices' must be either an integer or a list or a "
            f"cudf or dask_cudf Series or DataFrame, got: {type(vertices)}"
        )

    if isinstance(offsets, list):
        offsets = cudf.Series(offsets)

    if offsets is None:
        offsets = cudf.Series([0, len(vertices)])

    if not isinstance(offsets, cudf.Series):
        raise TypeError(
            f"'offsets' must be either 'None', a list or a "
            f"cudf Series, got: {type(offsets)}"
        )

    # vertices uses "external" vertex IDs, but since the graph has been
    # renumbered, the node ID must also be renumbered.
    if input_graph.renumbered:
        vertices = input_graph.lookup_internal_vertex_id(vertices)
        vertices_type = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    else:
        vertices_type = input_graph.input_df.dtypes.iloc[0]

    if isinstance(vertices, (cudf.Series, cudf.DataFrame)):
        vertices = dask_cudf.from_cudf(vertices, npartitions=input_graph._npartitions)
    vertices = vertices.astype(vertices_type)

    vertices = persist_dask_df_equal_parts_per_worker(
        vertices, client, return_type="dict"
    )

    do_expensive_check = False

    result = [
        client.submit(
            _call_induced_subgraph,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            vertices_[0] if vertices_ else cudf.Series(dtype=vertices_type),
            offsets,
            do_expensive_check,
            workers=[w],
            allow_other_workers=False,
        )
        for w, vertices_ in vertices.items()
    ]
    wait(result)

    cudf_result = [client.submit(convert_to_cudf, cp_arrays) for cp_arrays in result]

    wait(cudf_result)

    ddf = dask_cudf.from_delayed(cudf_result).persist()
    wait(ddf)

    if len(ddf) == 0:
        return None, None

    wait([(r.release(), c_r.release()) for r, c_r in zip(result, cudf_result)])

    ddf = ddf.sort_values("labels")

    # extract offsets from segmented induced_subgraph dataframes
    offsets = ddf["labels"].value_counts().compute().sort_index()
    offsets = cudf.concat([cudf.Series(0), offsets])
    offsets = (
        dask_cudf.from_cudf(
            offsets, npartitions=min(input_graph._npartitions, len(vertices))
        )
        .cumsum()
        .astype(vertices_type)
    )

    ddf = ddf.drop(columns="labels")

    if input_graph.renumbered:
        ddf = input_graph.unrenumber(ddf, "src")
        ddf = input_graph.unrenumber(ddf, "dst")

    return ddf, offsets
