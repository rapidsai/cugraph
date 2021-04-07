from cugraph.sampling import random_walks
from cugraph.community.egonet import batched_ego_graphs 
import cudf
import cugraph
from scipy import sparse
from collections import defaultdict

df = cudf.read_csv('./datasets/netscience.csv', delimiter=' ',
                  dtype=['int32', 'int32', 'float32'], header=None)
G = cugraph.Graph()
G.from_cudf_edgelist(df, source='0', destination='1',
                         edge_attr='2', renumber=False)

df, offsets = random_walks(G, [1, 4, 6],3)
#df, offsets = batched_ego_graphs(G, cudf.Series([1,4]),1)
print(df, "\n\n")
print(offsets)