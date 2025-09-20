# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import warnings
from cugraph.structure import Graph
import cudf
import numpy as np

from pylibcugraph import (
    force_atlas2 as plc_force_atlas2,
    ResourceHandle,
)


def renumber_vertices(input_graph, input_df, num_data_cols=1):
    if len(input_graph.renumber_map.implementation.col_names) > 1:
        cols = input_df.columns[:-num_data_cols].to_list()
    else:
        cols = "vertex"
    input_df = input_graph.add_internal_vertex_id(input_df, "vertex", cols)
    return input_df


def ensure_float32_dtype(input_series, input_series_name):
    if input_series.dtype != np.float32:
        warning_msg = (
            f"force_atlas2 requires '{input_series_name}' dtype to be "
            f"float32, but it is of type {input_series.dtype}. "
            f"Converting '{input_series_name}' to float32."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=3)
        input_series = input_series.astype(np.float32)
    return input_series


def ensure_vertex_dtype(input_graph, input_series, input_series_name):
    vertex_dtype = input_graph.edgelist.edgelist_df.dtypes.iloc[0]
    if input_series.dtype != vertex_dtype:
        warning_msg = (
            f"force_atlas2 requires '{input_series_name}' to match "
            "the graph's 'vertex' type. The input graph's vertex type is: "
            f"{vertex_dtype} and got '{input_series_name}' of type: "
            f"'{input_series.dtype}'. Converting."
        )
        warnings.warn(warning_msg, UserWarning, stacklevel=3)
        input_series = input_series.astype(vertex_dtype)
    return input_series


def force_atlas2(
    input_graph: Graph,
    max_iter=500,
    pos_list=None,
    outbound_attraction_distribution=True,
    lin_log_mode=False,
    prevent_overlapping=False,
    vertex_radius=None,
    overlap_scaling_ratio=100.0,
    edge_weight_influence=1.0,
    jitter_tolerance=1.0,
    barnes_hut_optimize=True,
    barnes_hut_theta=0.5,
    scaling_ratio=2.0,
    strong_gravity_mode=False,
    gravity=1.0,
    mobility=None,
    verbose=False,
    callback=None,
    *,
    random_state=None,
):

    """
    ForceAtlas2 is a continuous graph layout algorithm for handy network
    visualization.

    NOTE: Peak memory allocation occurs at 30*V.

    Parameters
    ----------
    input_graph : cugraph.Graph
        cuGraph graph descriptor with connectivity information.
        Edge weights, if present, should be single or double precision
        floating point values.

    max_iter : integer, optional (default=500)
        This controls the maximum number of levels/iterations of the Force
        Atlas algorithm. When specified the algorithm will terminate after
        no more than the specified number of iterations.
        No error occurs when the algorithm terminates in this manner.
        Good short-term quality can be achieved with 50-100 iterations.
        Above 1000 iterations is discouraged.

    pos_list: cudf.DataFrame, optional (default=None)
        Data frame with initial vertex positions containing three columns:
        'vertex', 'x' and 'y' positions.

    outbound_attraction_distribution: bool, optional (default=True)
        Distributes attraction along outbound edges.
        Hubs attract less and thus are pushed to the borders.

    lin_log_mode: bool, optional (default=False)
        Switch Force Atlas model from lin-lin to lin-log.
        Makes clusters more tight.

    prevent_overlapping: bool, optional (default=False)
        Prevent nodes to overlap.

    vertex_radius: cudf.DataFrame, optional (default=None)
        Data frame containing the radius of each vertex in the graph.
        Used only when prevent_overlapping is set to True.
        Must contain two columns 'vertex' and 'radius'.

    overlap_scaling_ratio: float, optional (default=100.0)
        Scaling of the repulsion force when two nodes are overlapping.
        Used only when prevent_overlapping is set to True.

    edge_weight_influence: float, optional (default=1.0)
        How much influence you give to the edges weight.
        0 is “no influence” and 1 is “normal”.

    jitter_tolerance: float, optional (default=1.0)
        How much swinging you allow. Above 1 discouraged.
        Lower gives less speed and more precision.

    barnes_hut_optimize: bool, optional (default=True)
        Whether to use the Barnes Hut approximation or the slower
        exact version.

    barnes_hut_theta: float, optional (default=0.5)
        Float between 0 and 1. Tradeoff for speed (1) vs
        accuracy (0) for Barnes Hut only.

    scaling_ratio: float, optional (default=2.0)
        How much repulsion you want. More makes a more sparse graph.
        Switching from regular mode to LinLog mode needs a readjustment
        of the scaling parameter.

    strong_gravity_mode: bool, optional (default=False)
        Sets a force that attracts the nodes that are distant from the
        center more. It is so strong that it can sometimes dominate other
        forces.

    gravity : float, optional (default=1.0)
        Attracts nodes to the center. Prevents islands from drifting away.

    mobility: cudf.DataFrame, optional (default=None)
        Data frame containing the mobility of each vertex in the graph.
        Mobility is a scaling factor on the speed of the vertex.
        Must contain two columns 'vertex' and 'mobility'.

    verbose: bool, optional (default=False)
        Output convergence info at each interation.

    callback: GraphBasedDimRedCallback, optional (default=None)
        .. versionremoved:: 25.10
            Support for the callback argument was removed in version 25.10.
            The last version of cugraph that supports it is 25.08.
            We apologize for not having a normal deprecation cycle.
            If you want this to be supported, please leave an issue:
            https://github.com/rapidsai/cugraph/issues

    random_state: int, optional (default=None)
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.

    Returns
    -------
    pos : cudf.DataFrame
        GPU data frame of size V containing three columns:
        the vertex identifiers and the x and y positions.

    Examples
    --------
    >>> from cugraph.datasets import karate
    >>> G = karate.get_graph(download=True)
    >>> pos = cugraph.force_atlas2(G)

    """

    if callback is not None:
        raise RuntimeError(
            "Support for the callback argument was removed in version 25.10. "
            "The last version of cugraph that supports it is 25.08. "
            "We apologize for not having a normal deprecation cycle. "
            "If you want this to be supported, please leave an issue: "
            "https://github.com/rapidsai/cugraph/issues"
        )
    initial_pos_vertices = None
    initial_pos_x = None
    initial_pos_y = None
    vertex_radius_vertices = None
    vertex_radius_values = None
    mobility_vertices = None
    mobility_values = None
    do_expensive_check = False

    if pos_list is not None:
        if not isinstance(pos_list, cudf.DataFrame):
            raise TypeError("pos_list should be a cudf.DataFrame")

        if set(pos_list.columns) != {"x", "y", "vertex"}:
            raise ValueError("pos_list has wrong column names")

        if input_graph.renumbered:
            pos_list = renumber_vertices(input_graph, pos_list, 2)
        # Ensure dtypes are valid, warn if we need to cast
        initial_pos_vertices = ensure_vertex_dtype(
            input_graph, pos_list["vertex"], 'pos_list["vertex"]'
        )
        initial_pos_x = ensure_float32_dtype(pos_list["x"], 'pos_list["x"]')
        initial_pos_y = ensure_float32_dtype(pos_list["y"], 'pos_list["y"]')

    if prevent_overlapping:
        if vertex_radius is None:
            raise ValueError(
                "vertex_radius must be provided when prevent_overlapping is enabled"
            )

        if not isinstance(vertex_radius, cudf.DataFrame):
            raise TypeError("vertex_radius must be a cudf.DataFrame")

        if set(vertex_radius.columns) != {"vertex", "radius"}:
            raise ValueError("vertex_radius has wrong column names")

        if input_graph.renumbered:
            vertex_radius = renumber_vertices(input_graph, vertex_radius)
        # Ensure dtypes are valid, warn if we need to cast
        vertex_radius_vertices = ensure_vertex_dtype(
            input_graph, vertex_radius["vertex"], 'vertex_radius["vertex"]'
        )
        vertex_radius_values = ensure_float32_dtype(
            vertex_radius["radius"], 'vertex_radius["radius"]'
        )

    if mobility is not None:
        if not isinstance(mobility, cudf.DataFrame):
            raise TypeError("mobility must be a cudf.DataFrame")

        if set(mobility.columns) != {"vertex", "mobility"}:
            raise ValueError("mobility has wrong column names")

        if input_graph.renumbered:
            mobility = renumber_vertices(input_graph, mobility)
        # Ensure dtypes are valid, warn if we need to cast
        mobility_vertices = ensure_vertex_dtype(
            input_graph, mobility["vertex"], 'mobility["vertex"]'
        )
        mobility_values = ensure_float32_dtype(
            mobility["mobility"], 'mobility["mobility"]'
        )

    if input_graph.is_directed():
        input_graph = input_graph.to_undirected()

    vertices, x_axis, y_axis = plc_force_atlas2(
        resource_handle=ResourceHandle(),
        random_state=random_state,
        graph=input_graph._plc_graph,
        max_iter=max_iter,
        start_vertices=initial_pos_vertices,
        x_start=initial_pos_x,
        y_start=initial_pos_y,
        outbound_attraction_distribution=outbound_attraction_distribution,
        lin_log_mode=lin_log_mode,
        prevent_overlapping=prevent_overlapping,
        vertex_radius_vertices=vertex_radius_vertices,
        vertex_radius_values=vertex_radius_values,
        overlap_scaling_ratio=overlap_scaling_ratio,
        edge_weight_influence=edge_weight_influence,
        jitter_tolerance=jitter_tolerance,
        barnes_hut_optimize=barnes_hut_optimize,
        barnes_hut_theta=barnes_hut_theta,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity_mode,
        gravity=gravity,
        mobility_vertices=mobility_vertices,
        mobility_values=mobility_values,
        verbose=verbose,
        do_expensive_check=do_expensive_check,
    )
    pos = cudf.DataFrame({"vertex": vertices, "x": x_axis, "y": y_axis})
    if input_graph.renumbered:
        pos = input_graph.unrenumber(pos, "vertex")
    return pos
