import logging
import time
from contextlib import contextmanager

import networkx as nx

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
    G = nx.readwrite.graphml.read_graphml("/datasets/rratzel/OSM/manhatten.graphml")
    #G = nx.readwrite.graphml.read_graphml("/datasets/rratzel/OSM/newyork.graphml")

print(f"{G.number_of_nodes()=}, {G.number_of_edges()=}")

# If nx-cugraph is being used, convert the graph separatly to ensure BC time is
# just for the algo. Do this as a "precache" operation since G must remain a NX
# graph to support the .nodes attr used below.
import os
if os.environ.get("NX_CUGRAPH_AUTOCONFIG") == "True":
    import nx_cugraph
    with timer("pre-caching"):
        Gcg = nx_cugraph.from_networkx(G)
        cache = G.__networkx_cache__.setdefault("backends", {}).setdefault("cugraph", {})
        key = nx.utils.backends._get_cache_key(
            edge_attrs=None,
            node_attrs=None,
            preserve_edge_attrs=False,
            preserve_node_attrs=False,
            preserve_graph_attrs=False,
        )
        nx.utils.backends._set_to_cache(cache, key, Gcg)

with timer("running BC"):
    betweenness_centrality = nx.betweenness_centrality(G)

betweenness_max_node = max(betweenness_centrality, key=betweenness_centrality.get)
y, x = G.nodes[betweenness_max_node]["y"], G.nodes[betweenness_max_node]["x"]

print(f"Betweeness Centrality intersection is at: {y},{x}")
