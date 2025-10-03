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

import time
import pytest
import random

import cudf
import cugraph
from cugraph.structure import number_map
from cugraph.internals import GraphBasedDimRedCallback
import scipy.io
from cugraph.datasets import (
    karate,
    polbooks,
    dolphins,
    netscience,
    dining_prefs,
)

# FIXME Removed the multi column positional due to it being non-deterministic
# need to replace this coverage. Issue 3890 in cuGraph repo was created.

# This method renumbers a dataframe so it can be tested using Trustworthiness.
# it converts a dataframe with string vertex ids to a renumbered int one.


def renumbered_edgelist(df):
    renumbered_df, num_map = number_map.NumberMap.renumber(df, "src", "dst")
    new_df = renumbered_df[["renumbered_src", "renumbered_dst", "wgt"]]
    column_names = {"renumbered_src": "src", "renumbered_dst": "dst"}
    new_df = new_df.rename(columns=column_names)
    return new_df


# This method converts a dataframe to a sparce matrix that is required by
# scipy Trustworthiness to verify the layout
def get_coo_array(edgelist):
    coo = edgelist
    x = max(coo["src"].max(), coo["dst"].max()) + 1
    row = coo["src"].to_numpy()
    col = coo["dst"].to_numpy()
    data = coo["wgt"].to_numpy()
    M = scipy.sparse.coo_array((data, (row, col)), shape=(x, x))

    return M


def cugraph_call(
    cu_M,
    max_iter,
    pos_list,
    outbound_attraction_distribution,
    lin_log_mode,
    prevent_overlapping,
    vertex_radius,
    overlap_scaling_ratio,
    edge_weight_influence,
    jitter_tolerance,
    barnes_hut_theta,
    barnes_hut_optimize,
    scaling_ratio,
    strong_gravity_mode,
    gravity,
    vertex_mobility,
    vertex_mass,
    callback=None,
    renumber=False,
):
    G = cugraph.Graph()
    if cu_M["src"] is not int or cu_M["dst"] is not int:
        renumber = True
    else:
        renumber = False
    G.from_cudf_edgelist(
        cu_M, source="src", destination="dst", edge_attr="wgt", renumber=renumber
    )

    t1 = time.time()
    pos = cugraph.force_atlas2(
        G,
        max_iter=max_iter,
        pos_list=pos_list,
        outbound_attraction_distribution=outbound_attraction_distribution,
        lin_log_mode=lin_log_mode,
        prevent_overlapping=prevent_overlapping,
        vertex_radius=vertex_radius,
        overlap_scaling_ratio=overlap_scaling_ratio,
        edge_weight_influence=edge_weight_influence,
        jitter_tolerance=jitter_tolerance,
        barnes_hut_optimize=barnes_hut_optimize,
        barnes_hut_theta=barnes_hut_theta,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity_mode,
        gravity=gravity,
        vertex_mobility=vertex_mobility,
        vertex_mass=vertex_mass,
        callback=callback,
    )
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))
    return pos


DATASETS = [
    (karate, 0.70),
    (polbooks, 0.75),
    (dolphins, 0.66),
    (netscience, 0.66),
    (dining_prefs, 0.4),
]

DATASETS2 = [
    (polbooks, 0.75),
    (dolphins, 0.66),
    (netscience, 0.66),
]

DATASETS_NOVERLAP = [
    (karate, 10.0, 50),
    (polbooks, 10.0, 90),
    (dolphins, 10.0, 60),
    (netscience, 10.0, 1100),
    (dining_prefs, 10.0, 20),
]

DATASETS_MOBILITY = [
    karate,
    polbooks,
    dolphins,
    netscience,
    dining_prefs,
]


MAX_ITERATIONS = [500]
BARNES_HUT_OPTIMIZE = [False, True]
MOBILITY_FIXED_CNT = [5, 12]


class ExampleCallback(GraphBasedDimRedCallback):
    def __init__(self):
        super().__init__()
        self.on_preprocess_end_called_count = 0
        self.on_epoch_end_called_count = 0
        self.on_train_end_called_count = 0

    def on_preprocess_end(self, positions):
        self.on_preprocess_end_called_count += 1

    def on_epoch_end(self, positions):
        self.on_epoch_end_called_count += 1

    def on_train_end(self, positions):
        self.on_train_end_called_count += 1


@pytest.mark.sg
@pytest.mark.parametrize("graph_file, score", DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("barnes_hut_optimize", BARNES_HUT_OPTIMIZE)
def test_force_atlas2(graph_file, score, max_iter, barnes_hut_optimize):
    cu_M = graph_file.get_edgelist(download=True)
    test_callback = ExampleCallback()
    with pytest.raises(RuntimeError, match="callback argument was removed"):
        cugraph_call(
            cu_M,
            max_iter=max_iter,
            pos_list=None,
            outbound_attraction_distribution=True,
            lin_log_mode=False,
            prevent_overlapping=False,
            vertex_radius=None,
            overlap_scaling_ratio=100.0,
            edge_weight_influence=1.0,
            jitter_tolerance=1.0,
            barnes_hut_optimize=False,
            barnes_hut_theta=0.5,
            scaling_ratio=2.0,
            strong_gravity_mode=False,
            gravity=1.0,
            vertex_mobility=None,
            vertex_mass=None,
            callback=test_callback,
        )
    """
        Trustworthiness score can be used for Force Atlas 2 as the algorithm
        optimizes modularity. The final layout will result in
        different communities being drawn out. We consider here the n x n
        adjacency matrix of the graph as an embedding of the nodes in high
        dimension. The results of force atlas 2 corresponds to the layout in
        a 2d space. Here we check that nodes belonging to the same community
        or neighbors are close to each other in the final embedding.
        Thresholds are based on the best score that is achived after 500
        iterations on a given graph.
    """
    # Uncomment the below if the callback becomes supported again
    """
    if "string" in graph_file.metadata["col_types"]:
        df = renumbered_edgelist(graph_file.get_edgelist(download=True))
        M = get_coo_array(df)
    else:
        M = get_coo_array(graph_file.get_edgelist(download=True))
    # from sklearn.manifold import trustworthiness
    cu_trust = trustworthiness(M, cu_pos[["x", "y"]].to_pandas())
    print(cu_trust, score)
    assert cu_trust > score
    # verify `on_preprocess_end` was only called once
    assert test_callback.on_preprocess_end_called_count == 1
    # verify `on_epoch_end` was called on each iteration
    assert test_callback.on_epoch_end_called_count == max_iter
    # verify `on_train_end` was only called once
    assert test_callback.on_train_end_called_count == 1
    """


@pytest.mark.sg
@pytest.mark.parametrize("graph_file, radius, max_overlaps", DATASETS_NOVERLAP)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
def test_force_atlas2_noverlap(graph_file, radius, max_overlaps, max_iter):
    """
    All vertices are given the same radius. After running FA2 with
    prevent_overlapping enabled, the number of pairs of overlapping
    vertices should be very low.
    Radii and thresholds have been picked such that
    99.9% of runs with prevent_overlapping would pass, and
    99.9% of runs without prevent_overlapping would fail
    """

    def count_overlaps(cu_pos, radius):
        pairs = cu_pos.merge(cu_pos, how="cross", suffixes=("_1", "_2"))
        pairs = pairs[pairs["vertex_1"] < pairs["vertex_2"]]
        # Calculate pairwise distances and find overlapping vertices
        overlaps = pairs[
            ((pairs["x_1"] - pairs["x_2"]) ** 2 + (pairs["y_1"] - pairs["y_2"]) ** 2)
            < (2 * radius) ** 2
        ]
        return len(overlaps)

    cu_M = graph_file.get_edgelist(download=True)
    G = graph_file.get_graph()
    vertex_radius = cudf.DataFrame(
        {
            "vertex": G.extract_vertex_list(return_unrenumbered_vertices=True),
            "radius": radius,
        }
    )

    cu_pos = cugraph_call(
        cu_M,
        max_iter=max_iter,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=True,
        vertex_radius=vertex_radius,
        overlap_scaling_ratio=100.0,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=False,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        vertex_mobility=None,
        vertex_mass=None,
        callback=None,
    )

    overlap_cnt = count_overlaps(cu_pos, radius)
    assert overlap_cnt < max_overlaps


@pytest.mark.sg
@pytest.mark.parametrize("graph_file", DATASETS_MOBILITY)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("fixed_node_cnt", MOBILITY_FIXED_CNT)
def test_force_atlas2_mobility(graph_file, max_iter, fixed_node_cnt):
    """
    After an initial layout, we freeze `fixed_node_cnt` random
    vertices with mobility=0.0, and run FA2 again with mobility enabled.
    In the final layout, the selected vertices should not have moved.
    """
    cu_M = graph_file.get_edgelist(download=True)
    G = graph_file.get_graph()
    vertices = G.extract_vertex_list(return_unrenumbered_vertices=True)

    mobility = [0.0] * fixed_node_cnt
    mobility += [1.0] * (len(vertices) - fixed_node_cnt)
    random.shuffle(mobility)

    mobility_df = cudf.DataFrame({"vertex": vertices, "mobility": mobility})

    # Initial layout without mobility
    pos_1 = cugraph_call(
        cu_M,
        max_iter=max_iter,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        vertex_radius=None,
        overlap_scaling_ratio=100.0,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=False,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        vertex_mobility=None,
        vertex_mass=None,
        callback=None,
    )

    # Run FA2 again, with mobility
    pos_2 = cugraph_call(
        cu_M,
        max_iter=max_iter,
        pos_list=pos_1,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        vertex_radius=None,
        overlap_scaling_ratio=100.0,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=False,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        vertex_mobility=mobility_df,
        vertex_mass=None,
        callback=None,
    )

    pos_difference = pos_1.merge(pos_2, on="vertex", suffixes=("_1", "_2")).merge(
        mobility_df, on="vertex", how="left"
    )

    pos_difference["move_dist"] = abs(
        pos_difference["x_1"] - pos_difference["x_2"]
    ) + abs(pos_difference["y_1"] - pos_difference["y_2"])

    fixed_nodes = pos_difference["mobility"] == 0.0
    assert fixed_nodes.sum() == fixed_node_cnt
    assert (pos_difference.loc[fixed_nodes, "move_dist"] == 0.0).all()
    assert (pos_difference.loc[~fixed_nodes, "move_dist"] > 0.0).all()


@pytest.mark.sg
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("barnes_hut_optimize", BARNES_HUT_OPTIMIZE)
def test_force_atlas2_mass(max_iter, barnes_hut_optimize):
    (v0, v1, v2, v3) = range(4)
    edge_dict = {
        "src": [v0, v0, v0, v1, v2, v3],
        "dst": [v1, v3, v2, v0, v0, v0],
        "wgt": 6 * [1.0],
    }
    edgelist_df = cudf.DataFrame(edge_dict)

    # v1, v2, v3 should be approximately equidistant from v0 and each other
    default_pos = cugraph_call(
        edgelist_df,
        max_iter=max_iter,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        vertex_radius=None,
        overlap_scaling_ratio=100.0,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=barnes_hut_optimize,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        vertex_mobility=None,
        vertex_mass=None,
        callback=None,
    )
    data = default_pos.set_index("vertex").sort_index().values.get()
    distances = scipy.spatial.distance.cdist(data, data)

    # Within 10% is close enough
    rel = 0.1
    # Equidistance from v0
    assert pytest.approx(distances[0, 1], rel=rel) == distances[0, 2]
    assert pytest.approx(distances[0, 1], rel=rel) == distances[0, 3]
    assert pytest.approx(distances[0, 2], rel=rel) == distances[0, 3]
    # Equidistance from each other (v1, v2, v3)
    assert pytest.approx(distances[1, 2], rel=rel) == distances[1, 3]
    assert pytest.approx(distances[1, 2], rel=rel) == distances[2, 3]
    assert pytest.approx(distances[1, 3], rel=rel) == distances[2, 3]

    # Now make v3 much larger (repels other vertices)
    vertex_mass = cudf.DataFrame({"vertex": [v0, v1, v2, v3], "mass": [4, 2, 2, 20.0]})
    vertex_mass["mass"] = vertex_mass["mass"].astype("float32")

    pos_2 = cugraph_call(
        edgelist_df,
        max_iter=max_iter,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        vertex_radius=None,
        overlap_scaling_ratio=100.0,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=barnes_hut_optimize,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        vertex_mobility=None,
        vertex_mass=vertex_mass,
        callback=None,
    )
    data = pos_2.set_index("vertex").sort_index().values.get()
    distances = scipy.spatial.distance.cdist(data, data)

    # v1 and v2 should be equidistance from v0
    assert pytest.approx(distances[0, 1], rel=rel) == distances[0, 2]
    # and from v3
    assert pytest.approx(distances[3, 1], rel=rel) == distances[3, 2]
    # v3 should be much further from v0 than v1 or v2
    assert distances[0, 3] / distances[0, 1] > 4.0
    assert distances[0, 3] / distances[0, 2] > 4.0
    assert distances[1, 3] / distances[1, 0] > 4.0
    assert distances[2, 3] / distances[2, 0] > 4.0
