# Copyright (c) 2019, NVIDIA CORPORATION.
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
from cugraph.tests import utils
from sklearn.manifold import trustworthiness
import scipy.io

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import fa2 and import networkx need to be
# relocated in the third-party group once this gets fixed.


def cugraph_call(cu_M, max_iter, pos_list, outbound_attraction_distribution,
                 lin_log_mode, prevent_overlapping, edge_weight_influence,
                 jitter_tolerance, barnes_hut_theta, barnes_hut_optimize,
                 scaling_ratio, strong_gravity_mode, gravity):

    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1', edge_attr='2',
                         renumber=False)

    # cugraph Force Atlas 2 Call
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
            gravity=gravity)
    t2 = time.time() - t1
    print('Cugraph Time : ' + str(t2))
    return pos


DATASETS = [('../datasets/karate.csv', 0.71),
            ('../datasets/polbooks.csv', 0.76),
            ('../datasets/dolphins.csv', 0.67),
            ('../datasets/netscience.csv', 0.67)]
MAX_ITERATIONS = [500]
BARNES_HUT_OPTIMIZE = [False, True]


@pytest.mark.parametrize('graph_file, score', DATASETS)
@pytest.mark.parametrize('max_iter', MAX_ITERATIONS)
@pytest.mark.parametrize('barnes_hut_optimize', BARNES_HUT_OPTIMIZE)
def test_force_atlas2(graph_file, score, max_iter,
                      barnes_hut_optimize):
    cu_M = utils.read_csv_file(graph_file)
    cu_pos = cugraph_call(cu_M,
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
                          gravity=1.0)
    '''
        Trustworthiness score can be used for Force Atlas 2 as the algorithm
        optimizes modularity. The final layout will result in
        different communities being drawn out. We consider here the n x n
        adjacency matrix of the graph as an embedding of the nodes in high
        dimension. The results of force atlas 2 corresponds to the layout in
        a 2d space. Here we check that nodes belonging to the same community
        or neighbors are close to each other in the final embedding.
        Thresholds are based on the best score that is achived after 500
        iterations on a given graph.
    '''

    matrix_file = graph_file[:-4] + '.mtx'
    M = scipy.io.mmread(matrix_file)
    M = M.todense()
    cu_trust = trustworthiness(M, cu_pos[['x', 'y']].to_pandas())
    assert cu_trust > score
