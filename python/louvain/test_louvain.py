import cugraph
import cudf
import numpy as np
import sys
import time
from scipy.io import mmread
import networkx as nx
import community
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
    values = cudf.Series(M.data)

    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, values)    

    # cugraph Louvain Call
    t1 = time.time()
    parts, mod = cugraph.nvLouvain(G)
    t2 =  time.time() - t1
    print('Time : '+str(t2))

    return parts, mod

def networkx_Call(M):
    M = M.tocsr()

    # Directed NetworkX graph
    Gnx = nx.Graph(M)

    #z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

    # Networkx Jaccard Call
    print('Solving... ')
    t1 = time.time()
    parts = community.best_partition(Gnx)
    t2 =  time.time() - t1

    print('Time : '+str(t2))
    return parts
   

datasets = ['/datasets/networks/karate.mtx', '/datasets/networks/dolphins.mtx', '/datasets/golden_data/graphs/dblp.mtx']

@pytest.mark.parametrize('graph_file', datasets)

def test_louvain(graph_file):
    M = ReadMtxFile(graph_file)
    cu_parts, cu_mod = cuGraph_Call(M)
    nx_parts = networkx_Call(M)
    
    # Calculating modularity scores for comparison
    Gnx = nx.Graph(M)
    cu_map = {0:0}
    for i in range(len(cu_parts)):
        cu_map[cu_parts['vertex'][i]] = cu_parts['partition'][i]
    cu_mod_nx = community.modularity(cu_map, Gnx)
    nx_mod = community.modularity(nx_parts, Gnx)
    assert len(cu_parts) == len(nx_parts)
    assert cu_mod > (.82 * nx_mod)
    assert abs(cu_mod - cu_mod_nx) < .0001