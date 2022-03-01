from networkx.algorithms.traversal import *
import cugraph
import cugraph.traversal.bfs as bfg
def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    return cugraph.bfs_edges(G,source,depth_limit)