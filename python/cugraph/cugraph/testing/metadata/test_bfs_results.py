import networkx as nx
import cugraph
from cugraph.experimental.datasets import dolphins, netscience, karate_disjoint
from cugraph.testing import utils

# for nx results related to test_bfs:

G_dolphins = utils.generate_nx_graph_from_file(dolphins.get_path())
G_netscience = utils.generate_nx_graph_from_file(netscience.get_path())
G_karate_dj = utils.generate_nx_graph_from_file(karate_disjoint.get_path())

# for nx results related to test_bfs_nonnative_inputs

# for nx results related to test_bfs_invalid_start