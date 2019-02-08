import cugraph
import cudf
import time
from scipy.io import mmread
import networkx as nx
import numpy as np
import pytest

print('Networkx version : {} '.format(nx.__version__))


def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()


def cugraph_Call(M, source):

    # Device data
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)

    print('sources size = ' + str(len(sources)))
    print('destinations size = ' + str(len(destinations)))

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, None)

    print('cugraph Solving... ')
    t1 = time.time()

    dist = cugraph.sssp(G, source)

    t2 = time.time() - t1
    print('Time : '+str(t2))

    distances = []
    for i, d in enumerate(dist['distance']):
        distances.append((i, d))

    return distances


def networkx_Call(M, source):

    print('Format conversion ... ')
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # should be autosorted, but check just to make sure
    if not M.has_sorted_indices:
        print('sort_indices ... ')
        M.sort_indices()

    # Directed NetworkX graph
    Gnx = nx.DiGraph(M)

    print('NX Solving... ')
    t1 = time.time()

    path=nx.single_source_shortest_path(Gnx, source)

    t2 = time.time() - t1

    print('Time : ' + str(t2))

    return path


datasets = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']

source = [1]


@pytest.mark.parametrize('graph_file', datasets)
@pytest.mark.parametrize('source', source)
def test_sssp(graph_file, source):

    M = ReadMtxFile(graph_file)
    cu_paths = cugraph_Call(M, source)
    nx_paths = networkx_Call(M, source)

    # Calculating mismatch
    err = 0

    for i in range(len(cu_paths)):
        if (cu_paths[i][1] != np.finfo(np.float32).max):
            if(cu_paths[i][1] != (len(nx_paths[cu_paths[i][0]])-1)):
                err = err + 1
        else:
            if (cu_paths[i][0] in nx_paths.keys()):
                err = err + 1

    assert err == 0
