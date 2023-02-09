# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import cudf
import cugraph
from cugraph.internals import GraphBasedDimRedCallback
from sklearn.manifold import trustworthiness
import scipy.io
from cugraph.experimental.datasets import karate, polbooks, dolphins, netscience

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import fa2 and import networkx need to be
# relocated in the third-party group once this gets fixed.


def cugraph_call(
    cu_M,
    max_iter,
    pos_list,
    outbound_attraction_distribution,
    lin_log_mode,
    prevent_overlapping,
    edge_weight_influence,
    jitter_tolerance,
    barnes_hut_theta,
    barnes_hut_optimize,
    scaling_ratio,
    strong_gravity_mode,
    gravity,
    callback=None,
):

    G = cugraph.Graph()
    G.from_cudf_edgelist(
        cu_M, source="src", destination="dst", edge_attr="wgt", renumber=False
    )

    t1 = time.time()
    pos = cugraph.force_atlas2(
        G,
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
        callback=callback,
    )
    t2 = time.time() - t1
    print("Cugraph Time : " + str(t2))
    return pos


DATASETS = [(karate, 0.70), (polbooks, 0.75), (dolphins, 0.66), (netscience, 0.66)]


MAX_ITERATIONS = [500]
BARNES_HUT_OPTIMIZE = [False, True]


class TestCallback(GraphBasedDimRedCallback):
    def __init__(self):
        super(TestCallback, self).__init__()
        self.on_preprocess_end_called_count = 0
        self.on_epoch_end_called_count = 0
        self.on_train_end_called_count = 0

    def on_preprocess_end(self, positions):
        self.on_preprocess_end_called_count += 1

    def on_epoch_end(self, positions):
        self.on_epoch_end_called_count += 1

    def on_train_end(self, positions):
        self.on_train_end_called_count += 1


@pytest.mark.parametrize("graph_file, score", DATASETS)
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("barnes_hut_optimize", BARNES_HUT_OPTIMIZE)
def test_force_atlas2(graph_file, score, max_iter, barnes_hut_optimize):
    cu_M = graph_file.get_edgelist()
    dataset_path = graph_file.get_path()
    test_callback = TestCallback()
    cu_pos = cugraph_call(
        cu_M,
        max_iter=max_iter,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
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

    matrix_file = dataset_path.with_suffix(".mtx")
    M = scipy.io.mmread(matrix_file)
    M = M.toarray()
    cu_trust = trustworthiness(M, cu_pos[["x", "y"]].to_pandas())
    print(cu_trust, score)
    assert cu_trust > score
    # verify `on_preprocess_end` was only called once
    assert test_callback.on_preprocess_end_called_count == 1
    # verify `on_epoch_end` was called on each iteration
    assert test_callback.on_epoch_end_called_count == max_iter
    # verify `on_train_end` was only called once
    assert test_callback.on_train_end_called_count == 1


# FIXME: this test occasionally fails - skipping to prevent CI failures but
# need to revisit ASAP
@pytest.mark.skip(reason="non-deterministric - needs fixing!")
@pytest.mark.parametrize("graph_file, score", DATASETS[:-1])
@pytest.mark.parametrize("max_iter", MAX_ITERATIONS)
@pytest.mark.parametrize("barnes_hut_optimize", BARNES_HUT_OPTIMIZE)
def test_force_atlas2_multi_column_pos_list(
    graph_file, score, max_iter, barnes_hut_optimize
):
    cu_M = graph_file.get_edgelist()
    dataset_path = graph_file.get_path()
    test_callback = TestCallback()
    pos = cugraph_call(
        cu_M,
        max_iter=max_iter,
        pos_list=None,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=False,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        callback=test_callback,
    )

    cu_M.rename(columns={"0": "src_0", "1": "dst_0"}, inplace=True)
    cu_M["src_1"] = cu_M["src_0"] + 1000
    cu_M["dst_1"] = cu_M["dst_0"] + 1000

    G = cugraph.Graph()
    G.from_cudf_edgelist(
        cu_M, source=["src_0", "src_1"], destination=["dst_0", "dst_1"], edge_attr="2"
    )

    pos_list = cudf.DataFrame()
    pos_list["vertex_0"] = pos["vertex"]
    pos_list["vertex_1"] = pos_list["vertex_0"] + 1000
    pos_list["x"] = pos["x"]
    pos_list["y"] = pos["y"]

    cu_pos = cugraph.force_atlas2(
        G,
        max_iter=max_iter,
        pos_list=pos_list,
        outbound_attraction_distribution=True,
        lin_log_mode=False,
        prevent_overlapping=False,
        edge_weight_influence=1.0,
        jitter_tolerance=1.0,
        barnes_hut_optimize=False,
        barnes_hut_theta=0.5,
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        callback=test_callback,
    )

    cu_pos = cu_pos.sort_values("0_vertex")
    matrix_file = dataset_path.with_suffix(".mtx")
    M = scipy.io.mmread(matrix_file)
    M = M.todense()
    cu_trust = trustworthiness(M, cu_pos[["x", "y"]].to_pandas())
    print(cu_trust, score)
    assert cu_trust > score
