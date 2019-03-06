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
    

def cuGraph_Call(M):
    M = M.tocsr()
    if M is None :
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    #Device data
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    
    G = cugraph.Graph()
    G.add_adj_list(row_offsets,col_indices,None)    

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.nvJaccard(G)
    t2 =  time.time() - t1
    print('Time : '+str(t2))

    return df['source'].to_array(), df['destination'].to_array(), df['jaccard_coeff'].to_array()

def networkx_Call(M):

    M = M.tocsr()
    M = M.tocoo()
    sources = M.row
    destinations = M.col
    edges = []
    for i in range(len(sources)):
        edges.append((sources[i],destinations[i]))  
    # in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this explicitly
    print('Format conversion ... ')

    # Directed NetworkX graph
    G = nx.DiGraph(M)
    Gnx = G.to_undirected()

    # Networkx Jaccard Call
    print('Solving... ')
    t1 = time.time()
    preds = nx.jaccard_coefficient(Gnx, edges)
    t2 =  time.time() - t1

    print('Time : '+str(t2))
    coeff = []
    src = []
    dst = []
    for u,v,p in preds:
        src.append(u)
        dst.append(v)
        coeff.append(p)
    return src, dst, coeff
   

datasets = ['/datasets/networks/dolphins.mtx', 
            '/datasets/networks/karate.mtx' , 
            '/datasets/networks/netscience.mtx']

@pytest.mark.parametrize('graph_file', datasets)

def test_jaccard(graph_file):

    M = ReadMtxFile(graph_file)
    cu_src, cu_dst, cu_coeff = cuGraph_Call(M)
    nx_src, nx_dst, nx_coeff = networkx_Call(M)
    # Calculating mismatch
    err = 0
    tol = 1.0e-06
    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if(abs(cu_coeff[i] -nx_coeff[i])>tol*1.1 and cu_src == nx_src and cu_dst == nx_dst):
            err+=1 
    print("Mismatches:  %d" %err)
    assert err == 0



