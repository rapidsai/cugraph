'''
Naming this with the _extension suffix allows it to be read by GaaS
as a graph creation extension.  This extension loads in the reddit
graph.
'''

import os
import numpy as np
import scipy.sparse as sp
import cudf
import cugraph
import networkx as nx
from cugraph.experimental import PropertyGraph

def load_reddit(data_dir):
    G = PropertyGraph()

    with open(os.path.join(data_dir, 'reddit_data.npz'), 'rb') as f:
        reddit_data = np.load(f)
        features = cudf.DataFrame(reddit_data["feature"], dtype='float32')
        features['id'] = cudf.Series(reddit_data['node_ids'], dtype='int32')
        features['y'] = cudf.Series(reddit_data['label'], dtype='int32')
        features['type'] = cudf.Series(reddit_data['node_types'], dtype='int32')
        features.columns = features.columns.astype('str')

        G.add_vertex_data(features, vertex_col_name='id')

    with open(os.path.join(data_dir, 'reddit_graph.npz'), 'rb') as f:
        M = sp.load_npz(f).tocsr()
        offsets = cudf.Series(M.indptr)
        indices = cudf.Series(M.indices)

        H = cugraph.Graph()
        H.from_cudf_adjlist(offsets, indices)
        G.add_edge_data(H.view_edge_list(), vertex_col_names=['src','dst'])

    return G