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
import networkx as nx
import os
import pytest

print ('Networkx version : {} '.format(nx.__version__))


def ReadMtxFile(mmFile):
    print('Reading '+ str(mmFile) + '...')
    return mmread(mmFile).asfptype()
    

def cuGraph_Callw(M):
   
    nnz_per_row = {r : 0 for r in range(M.get_shape()[0])}
    for nnz in range(M.getnnz()):
        nnz_per_row[M.row[nnz]] = 1 + nnz_per_row[M.row[nnz]]
    for nnz in range(M.getnnz()):
        M.data[nnz] = 1.0/float(nnz_per_row[M.row[nnz]])

    M = M.tocsr()
    if M is None :
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    #Device data
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    #values = cudf.Series(np.ones(len(col_indices), dtype = np.float32), nan_as_null = False)
    weights_arr = cudf.Series(np.ones(len(row_offsets), dtype = np.float32), nan_as_null = False)
    
    G = cugraph.Graph()
    G.add_adj_list(row_offsets,col_indices,None)    

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.nvJaccard_w(G, weights_arr)
    t2 =  time.time() - t1
    print('Time : '+str(t2))

    return df['jaccard_coeff']

   

datasets = ['/datasets/networks/dolphins.mtx', '/datasets/networks/karate.mtx', '/datasets/golden_data/graphs/dblp.mtx']

@pytest.mark.parametrize('graph_file', datasets)

def test_wjaccard(graph_file):

    M = ReadMtxFile(graph_file)
    cu_coeff = cuGraph_Callw(M)

    # no NetworkX equivalent to compare against...
    



