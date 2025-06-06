# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

from dask.distributed import wait, default_client
import dask_cudf
import cudf
import cupy as cp
import operator as op
from cugraph.dask.common.part_utils import (
    persist_dask_df_equal_parts_per_worker,
)

from pylibcugraph import ResourceHandle

from pylibcugraph import (
    biased_random_walks as pylibcugraph_biased_random_walks,
)

from cugraph.dask.comms import comms as Comms
from typing import Tuple, Union


def convert_to_cudf(
    cp_paths: cp.ndarray, number_map=None, is_vertex_paths: bool = False
) -> cudf.Series:
    """
    Creates cudf Series from cupy arrays from pylibcugraph wrapper
    """

    if is_vertex_paths and len(cp_paths) > 0:
        if number_map.implementation.numbered:
            df_ = cudf.DataFrame()
            df_["vertex_paths"] = cp_paths
            df_ = number_map.unrenumber(
                df_, "vertex_paths", preserve_order=True
            ).compute()
            vertex_paths = cudf.Series(df_["vertex_paths"]).fillna(-1)

            return vertex_paths

    return cudf.Series(cp_paths)


def _call_plc_biased_random_walks(
    sID: bytes, mg_graph_x, st_x: cudf.Series, max_depth: int, random_state: int
) -> Tuple[cp.ndarray, cp.ndarray]:

    return pylibcugraph_biased_random_walks(
        resource_handle=ResourceHandle(Comms.get_handle(sID).getHandle()),
        input_graph=mg_graph_x,
        start_vertices=st_x,
        max_length=max_depth,
        random_state=random_state,
    )


def biased_random_walks(
    input_graph,
    start_vertices: Union[int, list, cudf.Series, cudf.DataFrame, cudf.Series] = None,
    max_depth: int = 1,
    random_state: int = None,
) -> Tuple[Union[dask_cudf.Series, dask_cudf.DataFrame], dask_cudf.Series, int]:
    """
    compute random walks under the biased sampling framework for each nodes in
    'start_vertices' and returns a padded result along with the maximum path length.
    Vertices with no outgoing edges will be padded with -1.

    parameters
    ----------
    input_graph : cuGraph.Graph
        The graph can be either directed or undirected.

    start_vertices : int or list or cudf.Series or cudf.DataFrame
        A single node or a list or a cudf.Series of nodes from which to run
        the random walks. In case of multi-column vertices it should be
        a cudf.DataFrame

    max_depth: int
        The maximum depth of the random walks. If not specified, the maximum
        depth is set to 1.
        Must be a positive integer

    random_state: int, optional
        Random seed to use when making sampling calls.


    Returns
    -------
    vertex_paths : dask_cudf.Series or dask_cudf.DataFrame
        Series containing the vertices of edges/paths in the random walk.

    edge_weight_paths: dask_cudf.Series
        Series containing the edge weights of edges represented by the
        returned vertex_paths

    max_path_length : int
        The maximum path length
    """
    client = default_client()
    if isinstance(start_vertices, int):
        start_vertices = [start_vertices]

    if isinstance(start_vertices, list):
        start_vertices = cudf.Series(start_vertices)

    # start_vertices uses "external" vertex IDs, but if the graph has been
    # renumbered, the start vertex IDs must also be renumbered.
    if input_graph.renumbered:
        # FIXME: This should match start_vertices type to the renumbered df type
        # but verify that. If not retrieve the type and cast it when creating
        # the dask_cudf from a cudf
        start_vertices = input_graph.lookup_internal_vertex_id(start_vertices).compute()
        start_vertices_type = input_graph.edgelist.edgelist_df.dtypes[0]
    else:
        # FIXME: Get the 'src' column names instead and retrieve the type
        start_vertices_type = input_graph.input_df.dtypes.iloc[0]
    start_vertices = dask_cudf.from_cudf(
        start_vertices, npartitions=min(input_graph._npartitions, len(start_vertices))
    )
    start_vertices = start_vertices.astype(start_vertices_type)
    start_vertices = persist_dask_df_equal_parts_per_worker(
        start_vertices, client, return_type="dict"
    )

    result = [
        client.submit(
            _call_plc_biased_random_walks,
            Comms.get_session_id(),
            input_graph._plc_graph[w],
            start_v[0] if start_v else cudf.Series(dtype=start_vertices_type),
            max_depth,
            random_state=random_state,
            workers=[w],
            allow_other_workers=False,
        )
        for w, start_v in start_vertices.items()
    ]

    wait(result)

    result_vertex_paths = [client.submit(op.getitem, f, 0) for f in result]
    result_edge_wgt_paths = [client.submit(op.getitem, f, 1) for f in result]

    cudf_vertex_paths = [
        client.submit(convert_to_cudf, cp_vertex_paths, input_graph.renumber_map, True)
        for cp_vertex_paths in result_vertex_paths
    ]

    cudf_edge_wgt_paths = [
        client.submit(convert_to_cudf, cp_edge_wgt_paths)
        for cp_edge_wgt_paths in result_edge_wgt_paths
    ]

    wait([cudf_vertex_paths, cudf_edge_wgt_paths])

    ddf_vertex_paths = dask_cudf.from_delayed(cudf_vertex_paths).persist()
    ddf_edge_wgt_paths = dask_cudf.from_delayed(cudf_edge_wgt_paths).persist()
    wait([ddf_vertex_paths, ddf_edge_wgt_paths])

    # Wait until the inactive futures are released
    wait(
        [
            (r.release(), c_v.release(), c_e.release())
            for r, c_v, c_e in zip(result, cudf_vertex_paths, cudf_edge_wgt_paths)
        ]
    )

    return ddf_vertex_paths, ddf_edge_wgt_paths, max_depth
