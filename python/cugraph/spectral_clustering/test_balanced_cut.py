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

import cugraph
import cudf
import numpy as np
import sys
import time
from scipy.io import mmread
import community
import os
import pytest
import random

def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()

    
def cuGraph_Call(G, partitions):
    df = cugraph.spectralBalancedCutClustering(G, partitions, num_eigen_vects=partitions)
    score = cugraph.analyzeClustering_edge_cut(G, partitions, df['cluster'])
    return score

def random_Call(G, partitions):
    num_verts = G.num_vertices()
    assignment = []
    for i in range(num_verts):
        assignment.append(random.randint(0,partitions-1))
    assignment_cu = cudf.Series(assignment)
    score = cugraph.analyzeClustering_edge_cut(G, partitions, assignment_cu)
    return score
   

datasets = ['/datasets/networks/karate.mtx', '/datasets/networks/dolphins.mtx', '/datasets/golden_data/graphs/dblp.mtx']
partitions = [2, 4, 8]
@pytest.mark.parametrize('graph_file', datasets)
@pytest.mark.parametrize('partitions', partitions)
def test_modularityClustering(graph_file, partitions):
    # Read in the graph and get a cugraph object
    M = ReadMtxFile(graph_file).tocsr()
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    values = cudf.Series(M.data)
    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, values)
    
    # Get the modularity score for partitioning versus random assignment
    cu_score = cuGraph_Call(G, partitions)
    rand_score = random_Call(G, partitions)
    
    # Assert that the partitioning has better modularity than the random assignment
    assert cu_score < rand_score
