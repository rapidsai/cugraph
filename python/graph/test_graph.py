import cuGraph
import cudf
import pytest
import numpy as np
import networkx as nx
from scipy.io import mmread

print ('Networkx version : {} '.format(nx.__version__))


def ReadMtxFile(mmFile):
    print('Reading '+ str(mmFile) + '...')
    return mmread(mmFile).asfptype()

datasets = ['/datasets/networks/karate.mtx', '/datasets/golden_data/graphs/dblp.mtx']

@pytest.mark.parametrize('graph_file', datasets)

def test_add_edge_list_to_adj_list(graph_file):

    M = ReadMtxFile(graph_file)
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

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
	
    offsets_exp = cudf.Series(M.indptr)
    indices_exp = cudf.Series(M.indices)
    #values = cudf.Series(np.ones(len(sources), dtype = np.float64))

    # CuGraph add_egde_list to_adj_list call
    G = cuGraph.Graph()
    G.add_edge_list(sources,destinations, None)
    offsets, indices = G.to_adj_list()
    assert offsets == offsets_exp
    assert indices == indices_exp

@pytest.mark.parametrize('graph_file', datasets)

def test_add_adj_list_to_edge_list(graph_file):

    M = ReadMtxFile(graph_file)
    sources_exp = cudf.Series(M.row)
    destinations_exp = cudf.Series(M.col)

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
        
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    #values = cudf.Series(np.ones(len(sources), dtype = np.float64))

    # CuGraph add_adj_list to_edge_list call
    G = cuGraph.Graph()
    G.add_adj_list(offsets, indices, None)
    sources, destinations = G.to_edge_list()
    assert sources == sources_exp
    assert destinations == destinations_exp

@pytest.mark.parametrize('graph_file', datasets)

def test_delete_edge_list_delete_adj_list(graph_file):

    M = ReadMtxFile(graph_file)
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

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
        
    offsets = cudf.Series(M.indptr)
    indices = cudf.Series(M.indices)
    #values = cudf.Series(np.ones(len(sources), dtype = np.float64))

    # CuGraph delete_adj_list delete_edge_list call
    G = cuGraph.Graph()
    G.add_edge_list(sources, destinations, None)
    G.delete_edge_list()
    with pytest.raises(cudf.bindings.GDFError.GDFError) as excinfo:
        G.to_adj_list()
    assert excinfo.value.errcode.decode() == 'GDF_INVALID_API_CALL'

    G.add_adj_list(offsets, indices, None)
    G.delete_adj_list()
    with pytest.raises(cudf.bindings.GDFError.GDFError) as excinfo:
        G.to_edge_list()
    assert excinfo.value.errcode.decode() == 'GDF_UNSUPPORTED_METHOD'

