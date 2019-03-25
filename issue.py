import cugraph
import cudf
import sys, os
import time
import numpy as np	
from collections import OrderedDict
from scipy.io import mmread
import argparse

def read_mtx(mtx_file):
	M = mmread(mtx_file).asfptype()
	gdf = cudf.DataFrame() 
	gdf['src'] = cudf.Series(M.row)
	gdf['dst'] = cudf.Series(M.col)
	if M.data is None:
		gdf['val'] = 1.0
	else:
		gdf['val'] = cudf.Series(np.ones(len(M.data), dtype = np.float32))

	return gdf
def read_csv(csv_file):
	cols = ["src", "dst"]
	dtypes = OrderedDict([
	        ("src", "int32"), 
	        ("dst", "int32")
	        ])
	gdf = cudf.read_csv(csv_file, names=cols, delimiter='\t', dtype=list(dtypes.values()) )
	gdf['val'] = 1.0
	if gdf['val'].null_count > 0 :
		print("The reader failed to parse the input")
	if gdf['src'].null_count > 0 :
		print("The reader failed to parse the input")
	if gdf['dst'].null_count > 0 :
		print("The reader failed to parse the input")
	return gdf

#return the execution time 
def pagerank_call(G,alpha=0.85, max_iter=100, tol=1.0e-5):
	t1 = time.time()
	df = cugraph.pagerank(G,alpha,max_iter,tol)
	return time.time()-t1
def bfs_call(G,start=0):
	t1 = time.time()
	df = cugraph.bfs(G,start, True)
	return time.time()-t1
def sssp_call(G,start=0):
	t1 = time.time()
	df = cugraph.sssp(G,start)
	return time.time()-t1
def jaccard_call(G):
	t1 = time.time()
	df = cugraph.nvJaccard(G)
	print(df['jaccard_coeff'])
	return time.time()-t1
def louvain_call(G):
	#louvain requieres values in the graph
	t1 = time.time()
	df = cugraph.nvLouvain(G)
	return time.time()-t1

parser = argparse.ArgumentParser(description='CuGraph benchmark script.')
parser.add_argument('file', type=str,
                    help='Path to the input file')
parser.add_argument('--file_type', type=str, default="mtx",
                    help='Input file type : csv or mtx. If csv, cuDF reader is used (set for  [src dest] pairs separated by a tab). If mtx, Scipy reder is used (slow but supports weights). Default is mtx.')
parser.add_argument('--algo', type=str, default="all",
                    help='Algorithm to run : pagerank, bfs, sssp, jaccard, louvain, all. Default is all')
parser.add_argument('--damping_factor', type=float,default=0.85,
                    help='Damping factor for pagerank algo. Default is 0.85')
parser.add_argument('--max_iter', type=int, default=100,
                    help='Maximum number of iteration for any iterative algo. Default is 100')
parser.add_argument('--tolerence', type=float, default=1e-5,
                    help='Tolerence for any approximation algo. Default is 1e-5')
parser.add_argument('--source', type=int, default=0,
                    help='Source for bfs or sssp. Default is 0')
parser.add_argument('--auto_csr', type=int, default=0,
                    help='Automatically do the csr and transposed transformations. Default is 0, switch to another value to enable')
args = parser.parse_args()

# Load the data file
t1 = time.time()
if args.file_type == "mtx" :
	edgelist_gdf = read_mtx(args.file)
elif args.file_type == "csv" :
	edgelist_gdf = read_csv(args.file)
else:
	parser.error("File must be mtx or csv")
read_time = time.time()-t1
print(str(read_time), end="")

# create a Graph 
t1 = time.time()
G = cugraph.Graph()
G.add_edge_list(edgelist_gdf["src"], edgelist_gdf["dst"]) #, edgelist_gdf["val"])
if args.auto_csr == 0 :
	G.view_adj_list()
	G.view_transpose_adj_list()
cugraph_load_time = time.time()-t1
print(","+str(cugraph_load_time), end="")
# Call cugraph.pagerank to get the pagerank scores
if args.algo == "pagerank" or args.algo == "all":
	pagerank_time = pagerank_call(G, args.damping_factor, args.max_iter, args.tolerence)
	print(","+str(pagerank_time), end="")
if args.algo == "bfs" or args.algo == "all":
	bfs_time = bfs_call(G, args.source)
	print(","+str(bfs_time), end="")
if args.algo == "sssp" or args.algo == "all":
	#cugraph sssp curently prints stuff
	prev = sys.stdout
	sys.stdout = open(os.devnull, 'w')
	sssp_time = sssp_call(G, args.source)
	sys.stdout = prev
	print(","+str(sssp_time), end="")
if args.algo == "jaccard" or args.algo == "all":
	jaccard_time = jaccard_call(G)
	print(","+str(jaccard_time), end="")
if args.algo == "louvain" or args.algo == "all":
	louvain_time = louvain_call(G)
	print(","+str(louvain_time), end="")

print()

