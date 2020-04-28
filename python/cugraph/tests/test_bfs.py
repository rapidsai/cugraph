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
import queue
import time

import numpy as np
import pytest
import scipy
import cugraph
from cugraph.tests import utils
import rmm

# Temporarily suppress warnings till networkX fixes deprecation warnings
# (Using or importing the ABCs from 'collections' instead of from
# 'collections.abc' is deprecated, and in 3.8 it will stop working) for
# python 3.7.  Also, this import networkx needs to be relocated in the
# third-party group once this gets fixed.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import networkx as nx

def cugraph_call(cu_M, start_vertex):

    G = cugraph.DiGraph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1',
                         edge_attr='2')

    t1 = time.time()
    df = cugraph.bfs(G, start_vertex)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Return distances as np.array()
    return df['vertex'].to_array(), df['distance'].to_array()


def base_call(M, start_vertex):
    int_max = 2**31 - 1
    N = max(max(M['0']), max(M['1'])) + 1
    M = scipy.sparse.csr_matrix((M.weight, (M['0'], M['1'])),
                                shape=(N, N))

    offsets = M.indptr
    indices = M.indices
    num_verts = len(offsets) - 1
    dist = np.zeros(num_verts, dtype=np.int32)
    vertex = list(range(num_verts))

    for i in range(num_verts):
        dist[i] = int_max

    q = queue.Queue()
    q.put(start_vertex)
    dist[start_vertex] = 0
    while(not q.empty()):
        u = q.get()
        for i_col in range(offsets[u], offsets[u + 1]):
            v = indices[i_col]
            if (dist[v] == int_max):
                dist[v] = dist[u] + 1
                q.put(v)

    return vertex, dist

def cugraph_call_spc(G, start_vertex):

    t1 = time.time()
    df = cugraph.bfs(G, start_vertex, return_sp_counter=True)
    t2 = time.time() - t1
    #print('Time : '+str(t2))

    # Return distances as np.array()
    vertices = df['vertex'].to_array()
    sp_counter = df['sp_counter'].to_array()
    sp_counter_dict =  {vertices[idx]: sp_counter[idx] for idx in range(len(df))}
    return sp_counter_dict


def nx_call_spc(G, s):
    _, _, sigma = nx.networkx.algorithms.centrality.betweenness._single_source_shortest_path_basic(G, s)
    return sigma

DATASETS = ['../datasets/dolphins.csv',
            '../datasets/karate.csv',
            '../datasets/polbooks.csv',
            '../datasets/netscience.csv',
            '../datasets/email-Eu-core.csv']


# Test all combinations of default/managed and pooled/non-pooled allocation
#@pytest.mark.parametrize('managed, pool',
                         #list(product([False, True], [False, True])))
#@pytest.mark.parametrize('graph_file', DATASETS)
#def test_bfs(managed, pool, graph_file):
    #gc.collect()

    #rmm.reinitialize(
        #managed_memory=managed,
        #pool_allocator=pool,
        #initial_pool_size=2 << 27
    #)

    #assert(rmm.is_initialized())

    #M = utils.read_csv_for_nx(graph_file)
    #cu_M = utils.read_csv_file(graph_file)

    #base_vid, base_dist = base_call(M, 0)
    #cugraph_vid, cugraph_dist = cugraph_call(cu_M, 0)

    ## Calculating mismatch
    ## Currently, vertex order mismatch is not considered as an error
    #cugraph_idx = 0
    #base_idx = 0
    #distance_error_counter = 0
    #while cugraph_idx < len(cugraph_dist):
        #if base_vid[base_idx] == cugraph_vid[cugraph_idx]:
            ## An error is detected when for the same vertex
            ## the distances are different
            #if base_dist[base_idx] != cugraph_dist[cugraph_idx]:
                #distance_error_counter += 1
            #cugraph_idx += 1
        #base_idx += 1
    #assert distance_error_counter == 0

# ------------------------------------------------------------------------------
# Test for shortest path counting
def compare_close(result, expected, epsilon=1e-6):
  """
  """
  return np.isclose(result, expected, rtol=epsilon)#(result >= expected * (1.0 - epsilon)) and (result <= expected * (1.0 + epsilon))


SPC_CASE = [('../datasets/dolphins.csv', 10),
            ('../datasets/karate.csv', 5),
            ('../datasets/polbooks.csv', 2),
            ('../datasets/netscience.csv', 152),
            ('../datasets/email-Eu-core.csv', 200)]

SPC_CASE = [('../datasets/dolphins.csv', 10),
            ('../datasets/road_central.csv', 11116442),
            ('../datasets/road_central.csv', 1443588),
            ('../datasets/road_central.csv', 644832),
            ('../datasets/road_central.csv', 11598156)]

#SPC_CASE = [('../datasets/dolphins.csv', 10)]



##@pytest.mark.parametrize('managed, pool',
                         ##list(product([False, True], [False, True])))
#@pytest.mark.parametrize('managed, pool',
                         #list(product([False], [False])))
#@pytest.mark.parametrize('test_case', SPC_CASE)
#def test_bfs_spc(managed, pool, test_case):
    #""" Test BFS with shortest path counting (used for Betweenness Centrality)
    #"""
    #gc.collect()

    #rmm.reinitialize(
        #managed_memory=managed,
        #pool_allocator=pool,
        #initial_pool_size=2 << 27
    #)

    #assert(rmm.is_initialized())

    #graph_file, source = test_case

    #M = utils.read_csv_for_nx(graph_file)
    #Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  #create_using=nx.DiGraph())

    #cu_M = utils.read_csv_file(graph_file)
    #G = cugraph.DiGraph()
    #G.from_cudf_edgelist(cu_M, source='0', destination='1')


    #print("[DBG] Starting NX")
    #base_sp_counter = nx_call_spc(Gnx, source)
    #print("[DBG] Starting CU")
    #cugraph_sp_counter = cugraph_call_spc(G, source)

    ## Calculating mismatch
    ## Currently, vertex order mismatch is not considered as an error
    #cugraph_idx = 0
    #base_idx = 0
    #shortest_path_error_counter = 0
    ## Ensure that both are the same length
    #assert len(base_sp_counter) == len(cugraph_sp_counter), "Length mismatch"
    #missing_key_counter = 0
    #missmatch_sp_counter = 0
    ## Then check that each keys are in both
    ## TODO(xcadet): The problem is that the order is not the samee
    #for key in base_sp_counter:
        #if key in cugraph_sp_counter:
            #if not compare_close(cugraph_sp_counter[key], base_sp_counter[key]):
                #missing_key_counter += 1
                #print("[DBG][{}][{}] There is mismatch for vertex {}".format(graph_file, source, key))
        #else:
            #missing_key_counter += 1
            #print("[DBG][{}][{}] There is a missing key {}".format(graph_file, source, key))
    #assert missing_key_counter == 0, "Some keys were not found"
    #assert missmatch_sp_counter == 0, "Some shortest path counting were wrong"

##F_SPC_CASE = ['../datasets/dolphins.csv',
              ##'../datasets/netscience.csv']
#F_SPC_CASE = ['../datasets/dolphins.csv']
##F_SPC_CASE = ['../datasets/cti.csv']


##@pytest.mark.parametrize('managed, pool',
                         ##list(product([False, True], [False, True])))
#@pytest.mark.parametrize('managed, pool',
                         #list(product([False], [False])))
#@pytest.mark.parametrize('test_case', F_SPC_CASE)
#def test_full_bfs_spc(managed, pool, test_case):
    #""" Test BFS with shortest path counting (used for Betweenness Centrality)
    #"""
    #gc.collect()

    #rmm.reinitialize(
        #managed_memory=managed,
        #pool_allocator=pool,
        #initial_pool_size=2 << 27
    #)

    #assert(rmm.is_initialized())

    #graph_file = test_case

    #M = utils.read_csv_for_nx(graph_file)
    #Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  #create_using=nx.DiGraph())

    #cu_M = utils.read_csv_file(graph_file)
    #G = cugraph.DiGraph()
    #G.from_cudf_edgelist(cu_M, source='0', destination='1')

    #print("[DBG][NX]", len(Gnx.nodes()))
    #print("[DBG][NX]", len(Gnx.edges()))

    #print("[DBG][CU]", G.number_of_vertices())
    #print("[DBG][CU]", G.number_of_edges())


    #for source in Gnx:
        #base_sp_counter = nx_call_spc(Gnx, source)
        #cugraph_sp_counter = cugraph_call_spc(G, source)

        ## Calculating mismatch
        ## Currently, vertex order mismatch is not considered as an error
        #cugraph_idx = 0
        #base_idx = 0
        #shortest_path_error_counter = 0
        ## Ensure that both are the same length
        #assert len(base_sp_counter) == len(cugraph_sp_counter), "Length mismatch"
        #missing_key_counter = 0
        #missmatch_sp_counter = 0
        ## Then check that each keys are in both
        ## TODO(xcadet): The problem is that the order is not the samee
        #for key in base_sp_counter:
            #if key in cugraph_sp_counter:
                ## We are comparing floating point values
                #if not compare_close(cugraph_sp_counter[key], base_sp_counter[key]):
                    #missing_key_counter += 1
                    #print("[DBG][{}][{}] There is mismatch for vertex {}, cu {}, nx {}".format(graph_file, source, key, cugraph_sp_counter[key], base_sp_counter[key]))
                    #print("Key = {}".format(G.edgelist.renumber_map[G.edgelist.renumber_map == key].index[0]))
            #else:
                #missing_key_counter += 1
                #print("[DBG][{}][{}] There is a missing key {}".format(graph_file, source, key))
            #assert missing_key_counter == 0, "Some keys were not found"
            #assert missmatch_sp_counter == 0, "Some shortest path counting were wrong"

#===============================================================================
@pytest.mark.large
@pytest.mark.parametrize('managed, pool',
                         list(product([False], [False])))
@pytest.mark.parametrize('test_case', ["../datasets/cti.csv"])
def test_full_bfs_spc(managed, pool, test_case):
    """ Test BFS with shortest path counting (used for Betweenness Centrality)
    """
    gc.collect()

    rmm.reinitialize(
        managed_memory=managed,
        pool_allocator=pool,
        initial_pool_size=2 << 27
    )

    assert(rmm.is_initialized())

    graph_file = test_case

    M = utils.read_csv_for_nx(graph_file)
    Gnx = nx.from_pandas_edgelist(M, source='0', target='1',
                                  create_using=nx.Graph())

    cu_M = utils.read_csv_file(graph_file)
    G = cugraph.Graph()
    G.from_cudf_edgelist(cu_M, source='0', destination='1')

    print("[DBG][NX]", len(Gnx.nodes()))
    print("[DBG][NX]", len(Gnx.edges()))

    print("[DBG][CU]", G.number_of_vertices())
    print("[DBG][CU]", G.number_of_edges())


    for source in Gnx:#[10645]:
        print("[DBG] Processing source:", source)
        base_sp_counter = nx_call_spc(Gnx, source)
        cugraph_sp_counter = cugraph_call_spc(G, source)
        with open("/raid/xcadet/tmp/cu-renumber.txt".format(graph_file), "w") as out_fo:
            arr = G.edgelist.renumber_map.to_array()
            for idx in range(len(arr)):
                out_fo.write("{} <- {}\n".format(idx, arr[idx]))
        with open('/raid/xcadet/tmp/nx-bfs-{}.txt'.format(source), "w") as out_fo: # DBG
            for key in sorted(base_sp_counter.keys()):
                out_fo.write("{}\n".format(int(base_sp_counter[key])))
        with open('/raid/xcadet/tmp/cu-py-bfs-{}.txt'.format(source), "w") as out_fo: # DBG
            for key in sorted(cugraph_sp_counter.keys()):
                out_fo.write("{}\n".format(int(cugraph_sp_counter[key])))

        # Calculating mismatch
        # Currently, vertex order mismatch is not considered as an error
        cugraph_idx = 0
        base_idx = 0
        shortest_path_error_counter = 0
        # Ensure that both are the same length
        assert len(base_sp_counter) == len(cugraph_sp_counter), "Length mismatch"
        missing_key_counter = 0
        missmatch_sp_counter = 0
        # Then check that each keys are in both
        # TODO(xcadet): The problem is that the order is not the samee
        for key in base_sp_counter:
            if key in cugraph_sp_counter:
                # We are comparing floating point values
                if not compare_close(cugraph_sp_counter[key], base_sp_counter[key]):
                    missing_key_counter += 1
                    print("[DBG][{}][{}] There is mismatch for vertex {}, cu {}, nx {}".format(graph_file, source, key, cugraph_sp_counter[key], base_sp_counter[key]))
                    print("Key = {}".format(G.edgelist.renumber_map[G.edgelist.renumber_map == key].index[0]))
            else:
                missing_key_counter += 1
                print("[DBG][{}][{}] There is a missing key {}".format(graph_file, source, key))
            assert missing_key_counter == 0, "Some keys were not found"
            assert missmatch_sp_counter == 0, "Some shortest path counting were wrong"