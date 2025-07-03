import logging
import time
from contextlib import contextmanager

import networkx as nx
import cugraph

import cudf

################
import os
print(os.environ.get("NX_CUGRAPH_AUTOCONFIG"))

@contextmanager
def timer(message=""):
    if message: print(f"{message}...", end="", flush=True)
    st = time.perf_counter()
    yield
    print(f"done in {time.perf_counter() - st:.5f}s", flush=True)

print(f"Using NetworkX version {nx.__version__}")

logging.basicConfig(level=logging.DEBUG)

################

with timer("reading graphml"):
    G_nx = nx.readwrite.graphml.read_graphml("/datasets/rratzel/OSM/manhatten.graphml")
    # G = nx.readwrite.graphml.read_graphml("/datasets/rratzel/OSM/manhatten.graphml")
    #G = nx.readwrite.graphml.read_graphml("/datasets/rratzel/OSM/newyork.graphml")

print(f"{G_nx.number_of_nodes()=}, {G_nx.number_of_edges()=}")

# Convert NetworkX graph to cuGraph format
with timer("converting to cuGraphh"):
    # Create a DataFrame for edges
    edges = []
    for u, v in G_nx.edges():
        edges.append((int(u), int(v)))
    edges_df = cudf.DataFrame(edges, columns=["src", "dst"])

    # cuGraph requires integer node IDs
    G_cugraph = cugraph.Graph(directed=True)
    G_cugraph.from_cudf_edgelist(edges_df, source="src", destination="dst")

with timer("running BC"):
    bc_df = cugraph.betweenness_centrality(G_cugraph)

# Find node with highest centrality
max_node_row = bc_df.sort_values("betweenness_centrality", ascending=False).head(1)
max_node_id = int(max_node_row["vertex"].iloc[0])

# Map back to NetworkX for coordinates
y = G_nx.nodes[str(max_node_id)]["y"]
x = G_nx.nodes[str(max_node_id)]["x"]

print(f"Betweeness Centrality intersection is at: {y},{x}")
