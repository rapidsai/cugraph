from tempfile import NamedTemporaryFile
import random

# import pandas as pd
# import cupy as cp
# import numpy as np

# import cudf
import networkx as nx
import cugraph
from cugraph.experimental.datasets import dolphins, netscience, karate_disjoint, karate, polbooks
from cugraph.testing import utils
# from cugraph.experimental import datasets


# =============================================================================
# Parameters
# =============================================================================
SEEDS = [42]

DIRECTED_GRAPH_OPTIONS = [True, False]

DEPTH_LIMITS = [None, 1, 5, 18]

DATASETS = [dolphins, netscience, karate_disjoint]

#DATASETS = [dolphins, netscience, karate_disjoint, karate]

DATASETS_SMALL = [karate, dolphins, polbooks]

# tests/travesal/test_bfs:
test_bfs_results = {}
test_bfs_starts = {}

for ds in DATASETS + [karate]:
    for seed in SEEDS:
        for depth_limit in DEPTH_LIMITS:
            for dirctd in DIRECTED_GRAPH_OPTIONS:
                # this does the work of get_cu_graph_nx_results_and_params
                # print("seed:{}-depth_limit:{}-ds:{}-dirctd:{}".format(seed, depth_limit, ds, dirctd))
                Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=dirctd)

                random.seed(seed)
                start_vertex = random.sample(list(Gnx.nodes()), 1)[0]
                nx_values = nx.single_source_shortest_path_length(
                    Gnx, start_vertex, cutoff=depth_limit
                )

                test_bfs_results[str("{},{},{},{}".format(seed, depth_limit, ds, dirctd))] = nx_values
    test_bfs_starts[str("{},{}".format(seed, ds))] = start_vertex


# tests/traversal/test_sssp.py
test_sssp_results = {}

SOURCES = [1]

# [ParameterSet(values=([<cugraph.experimental.datasets.dataset.Dataset object at 0x7f0dc6f80ee0>, 1],), marks=[], id='ds:karate-src:1'), ParameterSet(values=([<cugraph.experimental.datasets.dataset.Dataset object at 0x7f0dc6f81d50>, 1],), marks=[], id='ds:dolphins-src:1'), ParameterSet(values=([<cugraph.experimental.datasets.dataset.Dataset object at 0x7f0dc6f81ed0>, 1],), marks=[], id='ds:polbooks-src:1')]

for ds in DATASETS_SMALL:
    for source in SOURCES:
        # directed=True
        Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=True)

        # At this moment, there is no support on if edgevals=False
        nx_paths = nx.single_source_dijkstra_path_length(Gnx, source)
        test_sssp_results[str("{},{}").format(ds, source)] = nx_paths
        # print("ds:{}".format(ds))


# tests/traversal/test_paths.py
CONNECTED_GRAPH = """1,5,3
1,4,1
1,2,1
1,6,2
1,7,2
4,5,1
2,3,1
7,6,2
"""

DISCONNECTED_GRAPH = CONNECTED_GRAPH + "8,9,4"

paths = [("1", "1"), ("1", "5"), ("1", "3"), ("1", "6")]
invalid_paths = {"connected": [("-1", "1"), ("0", "42")], "disconnected": [("1", "10"), ("1", "8")]}
#invalid_paths["connected"] = [("-1", "1"), ("0", "42")]
#invalid_paths["disconnected"] = [("1", "10"), ("1", "8")]

test_paths_results = {}

# CONNECTED_GRAPH
with NamedTemporaryFile(mode="w+", suffix=".csv") as graph_tf:
    graph_tf.writelines(CONNECTED_GRAPH)
    graph_tf.seek(0)
    Gnx = nx.read_weighted_edgelist(graph_tf.name, delimiter=",")

    graph_tf.writelines(DISCONNECTED_GRAPH)
    graph_tf.seek(0)
    Gnx_DIS = nx.read_weighted_edgelist(graph_tf.name, delimiter=",")

for path in paths:
    nx_path_length = nx.shortest_path_length(Gnx, path[0], target=path[1], weight="weight")
    cu_path_length = cugraph.shortest_path_length(Gnx, path[0], target=path[1])
    test_paths_results[str("{},{},{},nx").format(path[0], path[1], "connected")] = nx_path_length
    test_paths_results[str("{},{},{},cu").format(path[0], path[1], "connected")] = cu_path_length

# INVALID
for graph in ["connected", "disconnected"]:
    if graph == "connected":
        G = Gnx
    else:
        G = Gnx_DIS
    paths = invalid_paths[graph]
    for path in paths:
        try:
            test_paths_results[str("{},{},{},invalid").format(path[0], path[1], graph)] = cugraph.shortest_path_length(G, path[0], path[1])
        except ValueError:
            test_paths_results[str("{},{},{},invalid").format(path[0], path[1], graph)] = "ValueError"

# test_shortest_path_length_no_target
test_paths_results[str("1,notarget,nx")] = nx.shortest_path_length(Gnx_DIS, source="1", weight="weight")
test_paths_results[str("1,notarget,cu")] = cugraph.shortest_path_length(Gnx_DIS, "1")


# DEBUGGING
# print("test_bfs_results")
# print(test_bfs_results)
# print("test_bfs_starts")
# print(test_bfs_starts)
# print("test_sssp_results")
# print(test_sssp_results)
# print("test_paths_results")
# print(test_paths_results)

# RESULTS GENERATED FROM ABOVE CODE
# These golden results could be stored somewhere for the resultset class to access/grab
# test_bfs_results = {'42,None,dolphins,True': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1, 0: 2, 3: 2, 24: 2, 34: 2, 40: 2, 43: 2, 52: 2, 8: 2, 18: 2, 28: 2, 36: 2, 44: 2, 47: 2, 12: 2, 21: 2, 45: 2, 61: 2, 58: 2, 42: 2, 51: 2, 10: 3, 15: 3, 59: 3, 29: 3, 49: 3, 7: 3, 46: 3, 53: 3, 1: 3, 30: 3, 23: 3, 39: 3, 2: 3, 4: 3, 11: 3, 55: 3, 35: 4, 19: 4, 27: 4, 54: 4, 17: 4, 26: 4, 41: 4, 57: 4, 25: 5, 6: 5, 13: 5, 9: 5, 22: 5, 31: 5, 5: 5, 48: 5, 56: 6, 32: 6, 60: 7}, '42,None,dolphins,False': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1, 0: 2, 3: 2, 24: 2, 34: 2, 40: 2, 43: 2, 52: 2, 8: 2, 18: 2, 28: 2, 36: 2, 44: 2, 47: 2, 12: 2, 21: 2, 45: 2, 61: 2, 58: 2, 42: 2, 51: 2, 10: 3, 15: 3, 59: 3, 29: 3, 49: 3, 7: 3, 46: 3, 53: 3, 1: 3, 30: 3, 23: 3, 39: 3, 2: 3, 4: 3, 11: 3, 55: 3, 35: 4, 19: 4, 27: 4, 54: 4, 17: 4, 26: 4, 41: 4, 57: 4, 25: 5, 6: 5, 13: 5, 9: 5, 22: 5, 31: 5, 5: 5, 48: 5, 56: 6, 32: 6, 60: 7}, '42,1,dolphins,True': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1}, '42,1,dolphins,False': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1}, '42,5,dolphins,True': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1, 0: 2, 3: 2, 24: 2, 34: 2, 40: 2, 43: 2, 52: 2, 8: 2, 18: 2, 28: 2, 36: 2, 44: 2, 47: 2, 12: 2, 21: 2, 45: 2, 61: 2, 58: 2, 42: 2, 51: 2, 10: 3, 15: 3, 59: 3, 29: 3, 49: 3, 7: 3, 46: 3, 53: 3, 1: 3, 30: 3, 23: 3, 39: 3, 2: 3, 4: 3, 11: 3, 55: 3, 35: 4, 19: 4, 27: 4, 54: 4, 17: 4, 26: 4, 41: 4, 57: 4, 25: 5, 6: 5, 13: 5, 9: 5, 22: 5, 31: 5, 5: 5, 48: 5}, '42,5,dolphins,False': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1, 0: 2, 3: 2, 24: 2, 34: 2, 40: 2, 43: 2, 52: 2, 8: 2, 18: 2, 28: 2, 36: 2, 44: 2, 47: 2, 12: 2, 21: 2, 45: 2, 61: 2, 58: 2, 42: 2, 51: 2, 10: 3, 15: 3, 59: 3, 29: 3, 49: 3, 7: 3, 46: 3, 53: 3, 1: 3, 30: 3, 23: 3, 39: 3, 2: 3, 4: 3, 11: 3, 55: 3, 35: 4, 19: 4, 27: 4, 54: 4, 17: 4, 26: 4, 41: 4, 57: 4, 25: 5, 6: 5, 13: 5, 9: 5, 22: 5, 31: 5, 5: 5, 48: 5}, '42,18,dolphins,True': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1, 0: 2, 3: 2, 24: 2, 34: 2, 40: 2, 43: 2, 52: 2, 8: 2, 18: 2, 28: 2, 36: 2, 44: 2, 47: 2, 12: 2, 21: 2, 45: 2, 61: 2, 58: 2, 42: 2, 51: 2, 10: 3, 15: 3, 59: 3, 29: 3, 49: 3, 7: 3, 46: 3, 53: 3, 1: 3, 30: 3, 23: 3, 39: 3, 2: 3, 4: 3, 11: 3, 55: 3, 35: 4, 19: 4, 27: 4, 54: 4, 17: 4, 26: 4, 41: 4, 57: 4, 25: 5, 6: 5, 13: 5, 9: 5, 22: 5, 31: 5, 5: 5, 48: 5, 56: 6, 32: 6, 60: 7}, '42,18,dolphins,False': {16: 0, 14: 1, 20: 1, 33: 1, 37: 1, 38: 1, 50: 1, 0: 2, 3: 2, 24: 2, 34: 2, 40: 2, 43: 2, 52: 2, 8: 2, 18: 2, 28: 2, 36: 2, 44: 2, 47: 2, 12: 2, 21: 2, 45: 2, 61: 2, 58: 2, 42: 2, 51: 2, 10: 3, 15: 3, 59: 3, 29: 3, 49: 3, 7: 3, 46: 3, 53: 3, 1: 3, 30: 3, 23: 3, 39: 3, 2: 3, 4: 3, 11: 3, 55: 3, 35: 4, 19: 4, 27: 4, 54: 4, 17: 4, 26: 4, 41: 4, 57: 4, 25: 5, 6: 5, 13: 5, 9: 5, 22: 5, 31: 5, 5: 5, 48: 5, 56: 6, 32: 6, 60: 7}, '42,None,netscience,True': {1237: 0, 1238: 1}, '42,None,netscience,False': {1237: 0, 1238: 1}, '42,1,netscience,True': {1237: 0, 1238: 1}, '42,1,netscience,False': {1237: 0, 1238: 1}, '42,5,netscience,True': {1237: 0, 1238: 1}, '42,5,netscience,False': {1237: 0, 1238: 1}, '42,18,netscience,True': {1237: 0, 1238: 1}, '42,18,netscience,False': {1237: 0, 1238: 1}, '42,None,karate-disjoint,True': {19: 0, 0: 1, 1: 1, 33: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 21: 2, 31: 2, 30: 2, 9: 2, 14: 2, 15: 2, 18: 2, 20: 2, 22: 2, 23: 2, 26: 2, 27: 2, 28: 2, 29: 2, 32: 2, 16: 3, 24: 3, 25: 3}, '42,None,karate-disjoint,False': {19: 0, 0: 1, 1: 1, 33: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 21: 2, 31: 2, 30: 2, 9: 2, 14: 2, 15: 2, 18: 2, 20: 2, 22: 2, 23: 2, 26: 2, 27: 2, 28: 2, 29: 2, 32: 2, 16: 3, 24: 3, 25: 3}, '42,1,karate-disjoint,True': {19: 0, 0: 1, 1: 1, 33: 1}, '42,1,karate-disjoint,False': {19: 0, 0: 1, 1: 1, 33: 1}, '42,5,karate-disjoint,True': {19: 0, 0: 1, 1: 1, 33: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 21: 2, 31: 2, 30: 2, 9: 2, 14: 2, 15: 2, 18: 2, 20: 2, 22: 2, 23: 2, 26: 2, 27: 2, 28: 2, 29: 2, 32: 2, 16: 3, 24: 3, 25: 3}, '42,5,karate-disjoint,False': {19: 0, 0: 1, 1: 1, 33: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 21: 2, 31: 2, 30: 2, 9: 2, 14: 2, 15: 2, 18: 2, 20: 2, 22: 2, 23: 2, 26: 2, 27: 2, 28: 2, 29: 2, 32: 2, 16: 3, 24: 3, 25: 3}, '42,18,karate-disjoint,True': {19: 0, 0: 1, 1: 1, 33: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 21: 2, 31: 2, 30: 2, 9: 2, 14: 2, 15: 2, 18: 2, 20: 2, 22: 2, 23: 2, 26: 2, 27: 2, 28: 2, 29: 2, 32: 2, 16: 3, 24: 3, 25: 3}, '42,18,karate-disjoint,False': {19: 0, 0: 1, 1: 1, 33: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 21: 2, 31: 2, 30: 2, 9: 2, 14: 2, 15: 2, 18: 2, 20: 2, 22: 2, 23: 2, 26: 2, 27: 2, 28: 2, 29: 2, 32: 2, 16: 3, 24: 3, 25: 3}, '42,None,karate,True': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2, 30: 2, 9: 2, 27: 2, 28: 2, 32: 2, 16: 3, 33: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 4}, '42,None,karate,False': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2, 30: 2, 9: 2, 27: 2, 28: 2, 32: 2, 16: 3, 33: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 4}, '42,1,karate,True': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1}, '42,1,karate,False': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1}, '42,5,karate,True': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2, 30: 2, 9: 2, 27: 2, 28: 2, 32: 2, 16: 3, 33: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 4}, '42,5,karate,False': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2, 30: 2, 9: 2, 27: 2, 28: 2, 32: 2, 16: 3, 33: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 4}, '42,18,karate,True': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2, 30: 2, 9: 2, 27: 2, 28: 2, 32: 2, 16: 3, 33: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 4}, '42,18,karate,False': {7: 0, 0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 13: 2, 17: 2, 19: 2, 21: 2, 31: 2, 30: 2, 9: 2, 27: 2, 28: 2, 32: 2, 16: 3, 33: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 4}}
# test_bfs_starts = {'42,dolphins': 16, '42,netscience': 1237, '42,karate-disjoint': 19, '42,karate': 7}
# test_sssp_results = {'karate,1': {1: 0, 0: 1, 2: 1, 3: 1, 7: 1, 13: 1, 17: 1, 19: 1, 21: 1, 30: 1, 4: 2, 5: 2, 6: 2, 8: 2, 10: 2, 11: 2, 12: 2, 31: 2, 9: 2, 27: 2, 28: 2, 32: 2, 33: 2, 16: 3, 24: 3, 25: 3, 23: 3, 14: 3, 15: 3, 18: 3, 20: 3, 22: 3, 29: 3, 26: 3}, 'dolphins,1': {1: 0, 17: 1, 19: 1, 26: 1, 27: 1, 28: 1, 36: 1, 41: 1, 54: 1, 6: 2, 9: 2, 13: 2, 22: 2, 25: 2, 31: 2, 57: 2, 7: 2, 30: 2, 8: 2, 20: 2, 47: 2, 23: 2, 37: 2, 39: 2, 40: 2, 59: 2, 56: 3, 5: 3, 32: 3, 48: 3, 42: 3, 3: 3, 45: 3, 16: 3, 18: 3, 38: 3, 44: 3, 50: 3, 0: 3, 10: 3, 51: 3, 14: 3, 21: 3, 33: 3, 34: 3, 43: 3, 61: 3, 15: 3, 52: 3, 60: 4, 2: 4, 24: 4, 29: 4, 58: 4, 4: 4, 11: 4, 55: 4, 12: 4, 49: 4, 46: 4, 53: 4, 35: 5}, 'polbooks,1': {1: 0, 0: 1, 3: 1, 5: 1, 6: 1, 2: 2, 4: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 7: 2, 29: 2, 28: 3, 30: 3, 31: 3, 32: 3, 33: 3, 35: 3, 37: 3, 40: 3, 41: 3, 42: 3, 43: 3, 44: 3, 45: 3, 46: 3, 47: 3, 48: 3, 49: 3, 50: 3, 51: 3, 52: 3, 38: 3, 39: 3, 55: 3, 56: 3, 36: 3, 54: 3, 57: 3, 58: 3, 77: 3, 53: 3, 71: 3, 85: 3, 66: 4, 72: 4, 67: 4, 70: 4, 73: 4, 74: 4, 75: 4, 76: 4, 79: 4, 80: 4, 82: 4, 83: 4, 84: 4, 86: 4, 93: 4, 99: 4, 78: 4, 91: 4, 34: 4, 102: 4, 64: 4, 65: 4, 69: 4, 68: 4, 81: 4, 88: 5, 89: 5, 90: 5, 96: 5, 97: 5, 100: 5, 87: 5, 92: 5, 103: 5, 104: 5, 94: 5, 95: 5, 98: 5, 60: 5, 62: 5, 101: 5, 61: 5, 59: 5, 63: 5}}
# test_paths_results = {'1,1,connected,nx': 0, '1,1,connected,cu': 0.0, '1,5,connected,nx': 2.0, '1,5,connected,cu': 2.0, '1,3,connected,nx': 2.0, '1,3,connected,cu': 2.0, '1,6,connected,nx': 2.0, '1,6,connected,cu': 2.0, '-1,1,connected,invalid': 'ValueError', '0,42,connected,invalid': 'ValueError', '1,10,disconnected,invalid': 'ValueError'}

def get_bfs_results(test_params):
    return test_bfs_results[test_params]

def get_bfs_starts(test_params):
    return test_bfs_starts[test_params]

def get_sssp_results(test_params):
    return test_sssp_results[test_params]

def get_paths_results(test_params):
    return test_paths_results[test_params]
