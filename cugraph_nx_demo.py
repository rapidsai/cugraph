# Copyright (c) 2023, NVIDIA CORPORATION.
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


# NEW CELL

# Import needed libraries
import time
# import timeit
import cugraph
# import gc

import cugraph_nx as cnx
import networkx as nx
import pandas as pd

# import matplotlib.pyplot as plt

# import cudf
# import os
from cugraph.datasets import cyber

# NEW CELL

"""
try: 
    import matplotlib
except ModuleNotFoundError:
    os.system('pip install matplotlib')
"""

# ValueError from cyber's different source and destination column names
# from cugraph.datasets import cyber
# G = cyber.get_graph(download=True)

# NEW CELL

df = pd.read_csv("datasets/cyber.csv")
Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())

gdf = cyber.get_edgelist(download=True)
G = cugraph.from_cudf_edgelist(gdf, source="srcip", destination="dstip")
# Once dataset API can accept different src and dst col names, the above two lines can be replaced by the line below
# G = cyber.get_graph(download=True) 

# NEW CELL

# NEW CELL

# Defining preliminary performance tests
def cugraph_call_bc(G):
    t1 = time.time()
    cugraph.betweenness_centrality(G)
    t2 = time.time() - t1
    return t2

def cugraph_nx_call_bc(Gnx):
    t1 = time.time()
    cnx.betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2

def cugraph_nx_convert_call_bc(Gnx):
    G = cnx.from_networkx(Gnx)
    t1 = time.time()
    cnx.betweenness_centrality(G)
    t2 = time.time() - t1
    return t2

def networkx_call_bc(Gnx):
    t1 = time.time()
    nx.betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2


def cugraph_call_ebc(G):
    t1 = time.time()
    cugraph.edge_betweenness_centrality(G)
    t2 = time.time() - t1
    return t2

def cugraph_nx_call_ebc(Gnx):
    t1 = time.time()
    cnx.edge_betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2

def cugraph_nx_convert_call_ebc(Gnx):
    G = cnx.from_networkx(Gnx)
    t1 = time.time()
    cnx.edge_betweenness_centrality(G)
    t2 = time.time() - t1
    return t2

def networkx_call_ebc(Gnx):
    t1 = time.time()
    nx.edge_betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2

# NEW CELL

# Original analysis Tues Aug 22
tnx_bc = networkx_call_bc(Gnx)
trapids_bc = cugraph_call_bc(G)
tcnx_bc = cugraph_nx_call_bc(Gnx)
tcnx_convert_bc = cugraph_nx_convert_call_bc(Gnx)

tnx_ebc = networkx_call_ebc(Gnx)
trapids_ebc = cugraph_call_ebc(G)
tcnx_ebc = cugraph_nx_call_ebc(Gnx)
tcnx_convert_ebc = cugraph_nx_convert_call_ebc(Gnx)

print("Betweenness Centrality: cuGraph ({}), networkX ({})".format(trapids_bc, tnx_bc))
print("Betweenness Centrality: cu-nx w/o convert ({}), cu-nx w/ convert ({})".format(tcnx_bc, tcnx_convert_bc))
print("Edge Betweenness Centrality: cuGraph ({}), cuGraph-nx ({}), networkX ({})".format(trapids_ebc, tcnx_ebc, tnx_ebc))
print("Edge Betweenness Centrality: cu-nx w/o convert ({}), cu-nx w/ convert ({})".format(tcnx_ebc, tcnx_convert_ebc))

# NEW CELL

def cu_read_n_call_bc():
    t1 = time.time()
    gdf = cyber.get_edgelist(download=True)
    G = cugraph.from_cudf_edgelist(gdf, source="srcip", destination="dstip")
    cugraph.betweenness_centrality(G)
    t2 = time.time() - t1
    return t2


def cu_nx_noconvert_read_n_call_bc():
    # Could be refactored to be cnx.from_pandas_edgelist
    t1 = time.time()
    df = pd.read_csv("datasets/cyber.csv")
    Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())
    cnx.betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2


def cu_nx_convert_read_n_call_bc():
    # Could be refactored to be cnx.from_pandas_edgelist
    t1 = time.time()
    df = pd.read_csv("datasets/cyber.csv")
    Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())
    G = cnx.from_networkx(Gnx, preserve_all_attrs=True)
    cnx.betweenness_centrality(G)
    t2 = time.time() - t1
    return t2


def nx_read_n_call_bc():
    t1 = time.time()
    df = pd.read_csv("datasets/cyber.csv")
    Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())
    nx.betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2


def cu_read_n_call_ebc():
    t1 = time.time()
    gdf = cyber.get_edgelist(download=True)
    G = cugraph.from_cudf_edgelist(gdf, source="srcip", destination="dstip")
    cugraph.edge_betweenness_centrality(G)
    t2 = time.time() - t1
    return t2


def cu_nx_noconvert_read_n_call_ebc():
    # Could be refactored to be cnx.from_pandas_edgelist
    t1 = time.time()
    df = pd.read_csv("datasets/cyber.csv")
    Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())
    cnx.edge_betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2


def cu_nx_convert_read_n_call_ebc():
    # Could be refactored to be cnx.from_pandas_edgelist
    t1 = time.time()
    df = pd.read_csv("datasets/cyber.csv")
    Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())
    G = cnx.from_networkx(Gnx, preserve_all_attrs=True)
    cnx.edge_betweenness_centrality(G)
    t2 = time.time() - t1
    return t2


def nx_read_n_call_ebc():
    t1 = time.time()
    df = pd.read_csv("datasets/cyber.csv")
    Gnx = nx.from_pandas_edgelist(df, source="srcip", target="dstip", create_using=nx.Graph())
    nx.edge_betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2

# NEW CELL

tnx_read_bc = nx_read_n_call_bc()
trapids_read_bc = cu_read_n_call_bc()
tcnx_read_bc = cu_nx_noconvert_read_n_call_bc()
tcnx_read_convert_bc = cu_nx_convert_read_n_call_bc()

tnx_read_ebc = nx_read_n_call_ebc()
trapids_read_ebc = cu_read_n_call_ebc()
tcnx_read_ebc = cu_nx_noconvert_read_n_call_ebc()
tcnx_read_convert_ebc = cu_nx_convert_read_n_call_ebc()

print()
print("Graph Creation + Betweenness: cuGraph ({}), networkX ({})".format(trapids_read_bc, tnx_read_bc))
print("Graph Creation + Betweenness: cu-nx w/o convert ({}), cu-nx w/ convert ({})".format(tcnx_read_bc, tcnx_read_convert_bc))
print("Graph Creation + Edge Betweenness: cuGraph ({}), networkX ({})".format(trapids_read_ebc, tnx_read_ebc))
print("Graph Creation + Edge Betweenness: cu-nx w/o convert ({}), cu-nx w/ convert ({})".format(tcnx_read_ebc, tcnx_read_convert_ebc))

# NEW CELL

# NEW CELL

# NEW CELL (caused a SegFault when running cu.betweenness on preferentialAttachment.mtx)

"""# Advanced mtx performance tests?
def cugraph_call(M):
    gdf = cudf.DataFrame()
    gdf['src'] = M.row
    gdf['dst'] = M.col
    G = cugraph.Graph(directed=True)
    # From UserWarning: Betweenness centrality expects the 'store_transposed' flag to be set to 'False' for optimal performance
    # during the graph creation
    G.from_cudf_edgelist(gdf, source='src', destination='dst', renumber=False, store_transposed=False)
    

    t1 = time.time()
    cugraph.betweenness_centrality(G)
    t2 = time.time() - t1
    return t2


def networkx_call(M):
    nnz_per_row = {r: 0 for r in range(M.get_shape()[0])}
    for nnz in range(M.getnnz()):
        nnz_per_row[M.row[nnz]] = 1 + nnz_per_row[M.row[nnz]]
    for nnz in range(M.getnnz()):
        M.data[nnz] = 1.0/float(nnz_per_row[M.row[nnz]])

    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    Gnx = nx.DiGraph(M)

    t1 = time.time()
    nx.betweenness_centrality(Gnx)
    t2 = time.time() - t1
    return t2"""

# NEW CELL

"""from scipy.io import mmread

# Data reader - the file format is MTX, so we will use the reader from SciPy
def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    M = mmread(mm_file).asfptype()
     
    return M

# Prepare data files    
data = {
    'preferentialAttachment' : 'notebooks/data/preferentialAttachment.mtx',
}
#    'caidaRouterLevel'       : 'notebooks/data/caidaRouterLevel.mtx',
#    'coAuthorsDBLP'          : 'notebooks/data/coAuthorsDBLP.mtx',
#    'dblp'                   : 'notebooks/data/dblp-2010.mtx',
#    'citationCiteseer'       : 'notebooks/data/citationCiteseer.mtx',
#    'coPapersDBLP'           : 'notebooks/data/coPapersDBLP.mtx',
#    'coPapersCiteseer'       : 'notebooks/data/coPapersCiteseer.mtx',
#    'as-Skitter'             : 'notebooks/data/as-Skitter.mtx'"""

# NEW CELL

"""# arrays to capture performance gains
time_cu = []
time_nx = []
time_sp = []
perf_nx = []
perf_sp = []
names = []

# init libraries by doing a simple task 
v = 'notebooks/data/preferentialAttachment.mtx'
M = read_mtx_file(v)
del M


for k,v in data.items():
    gc.collect()

    # Saved the file Name
    names.append(k)
    
    # read the data
    M = read_mtx_file(v)
    
    # call cuGraph - this will be the baseline
    print("Starting cugraph call")
    trapids = cugraph_call(M)
    print("Cugraph call finished: {}".format(trapids))
    time_cu.append(trapids)
    
    # Now call NetworkX
    print("Starting networkX call")
    tn = networkx_call(M)
    print("NetworkX call finished: {}".format(tn))
    speedUp = (tn / trapids)
    perf_nx.append(speedUp)
    time_nx.append(tn)
        
    print("cuGraph (" + str(trapids) + ")  Nx (" + str(tn) + ")" )
    del M"""
