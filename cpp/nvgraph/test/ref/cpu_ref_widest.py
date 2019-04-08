#!/usr/bin/python

# Generates widest path vector for the single source vertex to all other vertices using dijkstra-like algorithm

# Usage : python3 nvgraph_cpu_ref.py graph.mtx source_vertex
# This works with networkx 1.8.1 (default ubuntu package version in 14.04)
# http://networkx.github.io/documentation/networkx-1.8/

# Latest version is currenlty 1.11 in feb 2016
# https://networkx.github.io/documentation/latest/tutorial/index.html

#import numpy as np
import sys
import time
from scipy.io import mmread
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys

#modified widest
def _dijkstra_custom(G, source, get_weight, cutoff=None):
    G_succ = G.succ if G.is_directed() else G.adj
    width = {node: -sys.float_info.max for node in range(G.number_of_nodes())}  # dictionary of final distances
    width[source] = sys.float_info.max
    #seen = set()
    Qset = set([(source, 0)])
    while len(Qset) > 0:
        u, depth = Qset.pop()
        if cutoff:
            if cutoff < depth:
                continue
        #print "Looking at vertex ", u, ", depth = ", depth
        for v, e in G_succ[u].items():
            cost = get_weight(u, v, e)
            #print "Looking at vertex ", u, ", edge to ", v
            if cost is None:
                continue
            alt = max(width[v], min(width[u], cost))
            if alt > width[v]:
                width[v] = alt
                Qset.add((v, depth+1))
        #print "Updated QSET: ", Qset
    return width

def single_source_dijkstra_widest(G, source, cutoff=None,
                                       weight='weight'):
    if G.is_multigraph():
        get_weight = lambda u, v, data: min(
            eattr.get(weight, 1) for eattr in data.values())
    else:
        get_weight = lambda u, v, data: data.get(weight, 1)

    return _dijkstra_custom(G, source, get_weight, cutoff=cutoff)

print ('Networkx version : {} '.format(nx.__version__))

# Command line arguments
argc = len(sys.argv)
if argc<=2:
    print("Error: usage is : python3 nvgraph_cpu_ref.py graph.mtx source_vertex")
    sys.exit()
mmFile = sys.argv[1]
src = int(sys.argv[2])
print('Reading '+ str(mmFile) + '...')
#Read
M = mmread(mmFile).transpose()

if M is None :
    raise TypeError('Could not read the input graph')

# in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this explicitly
M = M.asfptype().tolil().tocsr()
if not M.has_sorted_indices:
    M.sort_indices()

# Directed NetworkX graph
Gnx = nx.DiGraph(M)

#widest
print('Solving... ')
t1 = time.time()
widest = single_source_dijkstra_widest(Gnx,source=src)
t2 =  time.time() - t1

print('Time : '+str(t2))
print('Writing result ... ')

# fill missing with DBL_MAX
bwidest = np.full(M.shape[0], -sys.float_info.max, dtype=np.float64)
for r in widest.keys():
    bwidest[r] = widest[r]
#print bwidest
# write binary
out_fname = os.path.splitext(os.path.basename(mmFile))[0] + '_T.widest_' + str(src) + '.bin'
bwidest.tofile(out_fname, "")
print ('Result is in the file: ' + out_fname)

# write text
#f = open('/tmp/ref_' + os.path.basename(mmFile) + '_widest.txt', 'w')
#f.write(str(widest.values()))

print('Done')
