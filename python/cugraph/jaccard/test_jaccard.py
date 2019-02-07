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
    
    G = cugraph.Graph()
    G.add_adj_list(row_offsets,col_indices,None)    

    # cugraph Jaccard Call
    t1 = time.time()
    df = cugraph.nvJaccard(G)
    t2 =  time.time() - t1
    print('Time : '+str(t2))

    return df['jaccard_coeff']

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

    #z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

    # Networkx Jaccard Call
    print('Solving... ')
    t1 = time.time()
    preds = nx.jaccard_coefficient(Gnx, edges)
    t2 =  time.time() - t1

    print('Time : '+str(t2))
    coeff = []
    for u,v,p in preds:
        coeff.append(p)
    return coeff
   

datasets = ['/datasets/networks/dolphins.mtx', '/datasets/networks/karate.mtx', '/datasets/golden_data/graphs/dblp.mtx']

@pytest.mark.parametrize('graph_file', datasets)

def test_jaccard(graph_file):

    M = ReadMtxFile(graph_file)
    cu_coeff = cuGraph_Call(M)
    nx_coeff = networkx_Call(M)
    # Calculating mismatch
    err = 0
    tol = 1.0e-06
    assert len(cu_coeff) == len(nx_coeff)
    for i in range(len(cu_coeff)):
        if(abs(cu_coeff[i] -nx_coeff[i])>tol*1.1):
            err+=1 
    print("Mismatches:  %d" %err)
    assert err == 0



