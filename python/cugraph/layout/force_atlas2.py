# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cugraph.layout import force_atlas2_wrapper
from cugraph.structure.graph import null_check


def force_atlas2(
    input_graph,
    max_iter=500,
    pos_list=None,
    outbound_attraction_distribution=True,
    lin_log_mode=False,
    prevent_overlapping=False,
    edge_weight_influence=1.0,
    jitter_tolerance=1.0,
    barnes_hut_optimize=True,
    barnes_hut_theta=0.5,
    scaling_ratio=2.0,
    strong_gravity_mode=False,
    gravity=1.0,
    verbose=False,
    callback=None,
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
        max_iter : integer
            This controls the maximum number of levels/iterations of the Force
            Atlas algorithm. When specified the algorithm will terminate after
            no more than the specified number of iterations.
            No error occurs when the algorithm terminates in this manner.
            Good short-term quality can be achieved with 50-100 iterations.
            Above 1000 iterations is discouraged.
        pos_list: cudf.DataFrame
            Data frame with initial vertex positions containing two columns:
            'x' and 'y' positions.
        outbound_attraction_distribution: bool
            Distributes attraction along outbound edges.
            Hubs attract less and thus are pushed to the borders.
        lin_log_mode: bool
            Switch Force Atlas model from lin-lin to lin-log.
            Makes clusters more tight.
        prevent_overlapping: bool
            Prevent nodes to overlap.
        edge_weight_influence: float
            How much influence you give to the edges weight.
            0 is “no influence” and 1 is “normal”.
        jitter_tolerance: float
            How much swinging you allow. Above 1 discouraged.
            Lower gives less speed and more precision.
        barnes_hut_optimize: bool
            Whether to use the Barnes Hut approximation or the slower
            exact version.
        barnes_hut_theta: float
            Float between 0 and 1. Tradeoff for speed (1) vs
            accuracy (0) for Barnes Hut only.
        scaling_ratio: float
            How much repulsion you want. More makes a more sparse graph.
            Switching from regular mode to LinLog mode needs a readjustment
            of the scaling parameter.
        strong_gravity_mode: bool
            Sets a force that attracts the nodes that are distant from the
            center more. It is so strong that it can sometimes dominate other
            forces.
        gravity : float
            Attracts nodes to the center. Prevents islands from drifting away.
        verbose: bool
            Output convergence info at each interation.
        callback: GraphBasedDimRedCallback
            An instance of GraphBasedDimRedCallback class to intercept
            the internal state of positions while they are being trained.

            Example of callback usage:
                from cugraph.internals import GraphBasedDimRedCallback
                    class CustomCallback(GraphBasedDimRedCallback):
                        def on_preprocess_end(self, positions):
                            print(positions.copy_to_host())
                        def on_epoch_end(self, positions):
                            print(positions.copy_to_host())
                        def on_train_end(self, positions):
                            print(positions.copy_to_host())

        Returns
        -------
        pos : cudf.DataFrame
            GPU data frame of size V containing three columns:
            the vertex identifiers and the x and y positions.
    """

    if pos_list is not None:
        null_check(pos_list["vertex"])
        null_check(pos_list["x"])
        null_check(pos_list["y"])
        if input_graph.renumbered is True:
            pos_list = input_graph.add_internal_vertex_id(pos_list,
                                                          "vertex",
                                                          "vertex")

    if prevent_overlapping:
        raise Exception("Feature not supported")

    if input_graph.is_directed():
        input_graph = input_graph.to_undirected()

    pos = force_atlas2_wrapper.force_atlas2(
        input_graph,
        max_iter=max_iter,
        pos_list=pos_list,
        outbound_attraction_distribution=outbound_attraction_distribution,
        lin_log_mode=lin_log_mode,
        prevent_overlapping=prevent_overlapping,
        edge_weight_influence=edge_weight_influence,
        jitter_tolerance=jitter_tolerance,
        barnes_hut_optimize=barnes_hut_optimize,
        barnes_hut_theta=barnes_hut_theta,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity_mode,
        gravity=gravity,
        verbose=verbose,
        callback=callback,
    )

    if input_graph.renumbered:
        pos = input_graph.unrenumber(pos, "vertex")

    return pos
