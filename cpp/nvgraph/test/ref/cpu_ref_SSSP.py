#!/usr/bin/python

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
import networkx as nx
import os

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
M = mmread(mmFile).asfptype().tolil()

if M is None :
    raise TypeError('Could not read the input graph')

# in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this explicitly
M = M.transpose().tocsr()
if not M.has_sorted_indices:
    M.sort_indices()

# Directed NetworkX graph
Gnx = nx.DiGraph(M)

#SSSP
print('Solving... ')
t1 = time.time()
sssp = nx.single_source_dijkstra_path_length(Gnx,source=src)
t2 =  time.time() - t1

print('Time : '+str(t2))
print('Writing result ... ')

# fill missing with DBL_MAX
bsssp = np.full(M.shape[0], sys.float_info.max, dtype=np.float64)
for r in sssp.keys():
    bsssp[r] = sssp[r]
# write binary
out_fname = os.path.splitext(os.path.basename(mmFile))[0] + '_T.sssp_' + str(src) + '.bin'
bsssp.tofile(out_fname, "")
print ('Result is in the file: ' + out_fname)

# write text
#f = open('/tmp/ref_' + os.path.basename(mmFile) + '_sssp.txt', 'w')
#f.write(str(sssp.values()))

print('Done')
