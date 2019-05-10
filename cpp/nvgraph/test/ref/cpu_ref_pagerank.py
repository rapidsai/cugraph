#!/usr/bin/python

# Usage : python3 nvgraph_cpu_ref.py graph.mtx alpha
# This will convert matrix values to default probabilities
# This will also write same matrix in CSC format and with dangling notes

#import numpy as np
import sys
import time
from scipy.io import mmread
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import os
#from test_pagerank import pagerank

print ('Networkx version : {} '.format(nx.__version__))

# Command line arguments
argc = len(sys.argv)
if argc<=2:
    print("Error: usage is : python3 cpu_ref_pagerank.py graph.mtx alpha")
    sys.exit()
mmFile = sys.argv[1]
alpha = float(sys.argv[2])
print('Reading '+ str(mmFile) + '...')
#Read
M = mmread(mmFile).asfptype()
nnz_per_row = {r : 0 for r in range(M.get_shape()[0])}
for nnz in range(M.getnnz()):
    nnz_per_row[M.row[nnz]] = 1 + nnz_per_row[M.row[nnz]]
for nnz in range(M.getnnz()):
    M.data[nnz] = 1.0/float(nnz_per_row[M.row[nnz]])


MT = M.transpose(True)
M = M.tocsr()
if M is None :
    raise TypeError('Could not read the input graph')
if M.shape[0] != M.shape[1]:
    raise TypeError('Shape is not square')

# should be autosorted, but check just to make sure
if not M.has_sorted_indices:
    print('sort_indices ... ')
    M.sort_indices()

n = M.shape[0]
dangling = [0]*n 
for row in range(n):
    if M.indptr[row] == M.indptr[row+1]:
        dangling[row] = 1
    else:
        pass #M.data[M.indptr[row]:M.indptr[row+1]] = [1.0/float(M.indptr[row+1] - M.indptr[row])]*(M.indptr[row+1] - M.indptr[row])
#MT.data = M.data

# in NVGRAPH tests we read as CSR and feed as CSC, so here we doing this explicitly
print('Format conversion ... ')

# Directed NetworkX graph
print (M.shape[0])
Gnx = nx.DiGraph(M)

z = {k: 1.0/M.shape[0] for k in range(M.shape[0])}

#SSSP
print('Solving... ')
t1 = time.time()
pr = nx.pagerank(Gnx, alpha=alpha, nstart = z, max_iter=5000, tol = 1e-10) #same parameters as in NVGRAPH
t2 =  time.time() - t1

print('Time : '+str(t2))
print('Writing result ... ')


'''
#raw rank results
# fill missing with DBL_MAX
bres = np.zeros(M.shape[0], dtype=np.float64)
for r in pr.keys():
    bres[r] = pr[r]

print len(pr.keys())
# write binary
out_fname = '/tmp/' + os.path.splitext(os.path.basename(mmFile))[0] + '_T.pagerank_' + str(alpha) + '.bin'
bres.tofile(out_fname, "")
print 'Result is in the file: ' + out_fname
'''

#Indexes
sorted_pr = [item[0] for item in sorted(pr.items(), key=lambda x: x[1])]
bres = np.array(sorted_pr, dtype = np.int32)
#print (bres)
out_fname = os.path.splitext(os.path.basename(mmFile))[0] + '_T.pagerank_idx_' + str(alpha) + '.bin'
bres.tofile(out_fname, "")
print ('Vertices index sorted by pageranks in file: ' + out_fname)
#Values
out_fname =  os.path.splitext(os.path.basename(mmFile))[0] + '_T.pagerank_val_' + str(alpha) + '.bin'
#print (np.array(sorted(pr.values()),  dtype = np.float64))
np.array(sorted(pr.values()),  dtype = np.float64).tofile(out_fname, "")
print ('Pagerank sorted values in file: ' + out_fname)

print ('Converting and Writing CSC')

b = open(os.path.splitext(os.path.basename(mmFile))[0] + '_T.mtx', "w")
b.write("%%MatrixMarket matrix coordinate real general\n")
b.write("%%NVAMG rhs\n")
b.write("{} {} {}\n".format(n, n, M.getnnz()))

for item in range(MT.getnnz()):
    b.write("{} {} {}\n".format(MT.row[item] + 1, MT.col[item] + 1, MT.data[item]))
for val in dangling:
    b.write(str(val) + "\n")        
b.close()
print ("Wrote CSC to the file: "+ os.path.splitext(os.path.basename(mmFile))[0] + '_T.mtx')

print('Done')
