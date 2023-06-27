import random
import json

import pandas as pd
import cupy as cp
import numpy as np

import cudf
import networkx as nx
import cugraph
from cugraph.experimental.datasets import dolphins, netscience, karate_disjoint
from cugraph.testing import utils
from cugraph.experimental import datasets


# =============================================================================
# Parameters
# =============================================================================

SEEDS = [42]

DIRECTED_GRAPH_OPTIONS = [True, False]

DEPTH_LIMITS = [None, 1, 5, 18]

DATASETS = [dolphins, netscience, karate_disjoint]

# for nx results related to test_bfs:
test_bfs_results = {}
test_bfs_starts = {}

for ds in DATASETS:
    for seed in SEEDS:
        for depth_limit in DEPTH_LIMITS:
        #for ds in DATASETS:
            for dirctd in DIRECTED_GRAPH_OPTIONS:
                # this does the work of get_cu_graph_nx_results_and_params
                print("seed:{}-depth_limit:{}-ds:{}-dirctd:{}".format(seed, depth_limit, ds, dirctd))
                Gnx = utils.generate_nx_graph_from_file(ds.get_path(), directed=dirctd)

                random.seed(seed)
                start_vertex = random.sample(list(Gnx.nodes()), 1)[0]
                nx_values = nx.single_source_shortest_path_length(
                    Gnx, start_vertex, cutoff=depth_limit
                )

                test_bfs_results[str("{},{},{},{}".format(seed, depth_limit, ds, dirctd))] = nx_values
                #test_bfs_starts[str("{},{},{},{}".format(seed, depth_limit, ds, dirctd))] = start_vertex
    test_bfs_starts[str("{},{}".format(seed, ds))] = start_vertex
                #print(nx_values)
                #breakpoint()

print("test_bfs_results")
print(test_bfs_results)
print("test_bfs_starts")
print(test_bfs_starts)

#test_bfs_results_saved = 

def get_bfs_results(test_params):
    return test_bfs_results[test_params]

def get_bfs_starts(test_params):
    return test_bfs_starts[test_params]

#G_dolphins = utils.generate_nx_graph_from_file(dolphins.get_path())
#G_netscience = utils.generate_nx_graph_from_file(netscience.get_path())
#G_karate_dj = utils.generate_nx_graph_from_file(karate_disjoint.get_path())


# for nx results related to test_bfs_nonnative_inputs

# for nx results related to test_bfs_invalid_start