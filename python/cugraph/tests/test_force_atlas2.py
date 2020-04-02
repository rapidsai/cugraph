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

import gc
from itertools import product
import time
import numpy as np

import pytest

import cudf
import cugraph
from cugraph.tests import utils
import rmm
from random import randint

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, these import fa2 and import networkx need to be
# relocated in the third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import fa2
    import networkx as nx


print('Networkx version : {} '.format(nx.__version__))


def cugraph_call(cu_M, max_iter, pos_list, outbound_attraction_distribution,
                 lin_log_mode, prevent_overlapping, edge_weight_influence,
                 jitter_tolerance, barnes_hut_theta, barnes_hut_optimize,
                 scaling_ratio, strong_gravity_mode, gravity):

    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')
    # cugraph Force Atlas 2 Call
    t1 = time.time()
    pos = cugraph.force_atlas2(G,
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


def networkx_call(M, max_iter,
                pos_list,
                node_masses,
                outbound_attraction_distribution,
                lin_log_mode,
                prevent_overlapping,
                edge_weight_influence,
                jitter_tolerance,
                barnes_hut_optimize,
                barnes_hut_theta,
                scaling_ratio,
                strong_gravity_mode,
                multithread,
                gravity):
    Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  edge_attr='weight', create_using=nx.Graph())
    # Networkx Force Atlas 2 Call
    print('Solving... ')
    t1 = time.time()
    forceatlas2 = fa2.ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=outbound_attraction_distribution,# Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=edge_weight_influence,

            # Performance
            jitterTolerance=jitter_tolerance,  # Tolerance
            barnesHutOptimize=barnes_hut_theta,
            barnesHutTheta=barnes_hut_optimize,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=scaling_ratio,
            strongGravityMode=scaling_ratio,
            gravity=gravity,

            # Log
            verbose=False)

    pos = forceatlas2.forceatlas2_networkx_layout(Gnx, pos=pos_list, iterations=max_iter)
    
    t2 = time.time() - t1

    print('Networkx Time : ' + str(t2))
    return pos


DATASETS = ['../datasets/karate.csv']
            #'../datasets/dolphins.csv']

MAX_ITERATIONS = [0]
OUTBOUND_ATTRACTION_DISTRIBUTION = [True]
EDGE_WEIGHT_INFLUENCE = [1.0]
JITTER_TOLERANCE = [1.0]
BARNES_HUT_THETA = [0.5]
SCALING_RATIO = [2.0]
STRONG_GRAVITY_MODE = [False]
GRAVITY = [1.0]

# Test all combinations of default/managed and pooled/non-pooled allocation
@pytest.mark.parametrize('managed, pool',
                         list(product([False, True], [False, True])))
@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('max_iter', MAX_ITERATIONS)
@pytest.mark.parametrize('outbound_attraction_distribution', OUTBOUND_ATTRACTION_DISTRIBUTION)
@pytest.mark.parametrize('edge_weight_influence', EDGE_WEIGHT_INFLUENCE)
@pytest.mark.parametrize('jitter_tolerance', JITTER_TOLERANCE)
@pytest.mark.parametrize('barnes_hut_theta', BARNES_HUT_THETA)
@pytest.mark.parametrize('scaling_ratio', SCALING_RATIO)
@pytest.mark.parametrize('strong_gravity_mode', STRONG_GRAVITY_MODE)
@pytest.mark.parametrize('gravity', GRAVITY)
def test_force_atlas2(managed, pool, graph_file, max_iter,
                      outbound_attraction_distribution, edge_weight_influence,
                      jitter_tolerance, barnes_hut_theta, scaling_ratio,
                      strong_gravity_mode, gravity):
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    M = utils.read_csv_for_nx(graph_file)
    cu_M = utils.read_csv_file(graph_file)

    #Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
    #                              edge_attr='weight', create_using=nx.Graph())

    Gnx = nx.OrderedGraph()
    for i in range(len(M)):
            Gnx.add_node(M['0'][i])
    for i in range(len(M)):
        Gnx.add_edge(M['0'][i], M['1'][i], weight=M['weight'][i])

    print(Gnx.nodes())
    # Init nodes at same positions

    nx_pos_list = dict()
    for node in Gnx:
        nx_pos_list[node] = randint(-100, 100), randint(-100, 100)

    print(nx_pos_list)
    k = np.fromiter(nx_pos_list.keys(), dtype='int32')
    x = np.fromiter([x for x, _ in nx_pos_list.values()], dtype='float32')
    y = np.fromiter([y for _, y in nx_pos_list.values()], dtype='float32')

    cu_pos_list = cudf.DataFrame({'vertex': k,
                                  'x': x,
                                  'y' : y}) 
 
    cu_pos = cugraph_call(cu_M,
                          max_iter=max_iter,
                          pos_list=cu_pos_list,
                          outbound_attraction_distribution=outbound_attraction_distribution,
                          lin_log_mode=False,
                          prevent_overlapping=False,
                          edge_weight_influence=edge_weight_influence,
                          jitter_tolerance=jitter_tolerance,
                          barnes_hut_optimize=False,
                          barnes_hut_theta=barnes_hut_theta,
                          scaling_ratio=scaling_ratio,
                          strong_gravity_mode=strong_gravity_mode,
                          gravity=gravity)

    nx_pos = networkx_call(M,
                           max_iter=max_iter,
                           pos_list=nx_pos_list,
                           node_masses=None,
                           outbound_attraction_distribution=outbound_attraction_distribution,
                           lin_log_mode=False,
                           prevent_overlapping=False,
                           edge_weight_influence=edge_weight_influence,
                           jitter_tolerance=jitter_tolerance,
                           barnes_hut_optimize=False,
                           barnes_hut_theta=barnes_hut_theta,
                           scaling_ratio=scaling_ratio,
                           strong_gravity_mode=strong_gravity_mode,
                           multithread=False,
                           gravity=gravity)
   
    print("Cu pos")
    print(cu_pos)
    print()
    print("Nx pos")
    print(nx_pos)
    # Check positions are the same
    assert len(cu_pos) == len(nx_pos)
    err = 0
    for i in range(len(cu_pos_list)):
        node = cu_pos_list['vertex'][i]
        if abs(cu_pos['x'][i] - nx_pos[node][0]) > 0.01 \
        and abs(cu_pos['y'][i] - nx_pos[node][1]) > 0.01:
            err += 1
    print("Mismatched points:", err)
    assert err < 0.01 * len(cu_pos)
