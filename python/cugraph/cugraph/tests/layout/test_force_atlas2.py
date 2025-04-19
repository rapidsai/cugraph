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

import cugraph
from cugraph.structure import number_map
from cugraph.internals import GraphBasedDimRedCallback
from sklearn.manifold import trustworthiness
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
    overlap_scaling_ratio
    edge_weight_influence,
    jitter_tolerance,
    barnes_hut_theta,
    barnes_hut_optimize,
    scaling_ratio,
    strong_gravity_mode,
    gravity,
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
    (dining_prefs, 0.50),
]

DATASETS2 = [
    (polbooks, 0.75),
    (dolphins, 0.66),
    (netscience, 0.66),
]


MAX_ITERATIONS = [500]
BARNES_HUT_OPTIMIZE = [False, True]


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
    cu_pos = cugraph_call(
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

    if "string" in graph_file.metadata["col_types"]:
        df = renumbered_edgelist(graph_file.get_edgelist(download=True))
        M = get_coo_array(df)
    else:
        M = get_coo_array(graph_file.get_edgelist(download=True))
    cu_trust = trustworthiness(M, cu_pos[["x", "y"]].to_pandas())
    print(cu_trust, score)
    assert cu_trust > score
    # verify `on_preprocess_end` was only called once
    assert test_callback.on_preprocess_end_called_count == 1
    # verify `on_epoch_end` was called on each iteration
    assert test_callback.on_epoch_end_called_count == max_iter
    # verify `on_train_end` was only called once
    assert test_callback.on_train_end_called_count == 1
