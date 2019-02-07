import cugraph
import cudf
import time
from scipy.io import mmread
import pytest
import numpy as np

def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()


def cugraph_Call(M, start_vertex):

    # Device data
    M = M.tocsr()
    sources = cudf.Series(M.indptr)
    destinations = cudf.Series(M.indices)
    values = cudf.Series(M.data)
    
    G = cugraph.Graph()
    G.add_adj_list(sources, destinations, values)
    
    t1 = time.time()
    df = cugraph.bfs(G, start_vertex)
    t2 = time.time() - t1
    print('Time : '+str(t2))

    # Return distances as np.array()
    return np.array(df['distance'])


def base_Call(M, start_vertex):
    intMax = 2147483647
    M = M.tocsr()
    offsets = M.indptr
    indices = M.indices
    num_verts = len(offsets) - 1
    dist = np.zeros(num_verts, dtype=np.int32)
    
    for i in range(num_verts):
        dist[i] = intMax
    import queue
    q = queue.Queue()
    q.put(start_vertex)
    dist[start_vertex] = 0
    while(not q.empty()):
        u = q.get()
        for iCol in range(offsets[u],offsets[u + 1]):
            v = indices[iCol]
            if (dist[v] == intMax):
                dist[v] = dist[u] + 1
                q.put(v)
    return dist

datasets = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']

@pytest.mark.parametrize('graph_file', datasets)
def test_bfs(graph_file):

    M = ReadMtxFile(graph_file)
    base_dist = base_Call(M, 0)
    dist = cugraph_Call(M, 0)
    
    assert len(base_dist) == len(dist)
    for i in range(len(dist)):
        assert base_dist[i] == dist[i]