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
# import time
import cugraph

# import cugraph_nx as cnx
import networkx as nx

# import cudf
# import os

# NEW CELL

# from cugraph.datasets import cyber
# G = cyber.get_graph(download=True) ERROR OCCURS FOR SOME REASON
#   File "...cugraph/structure/graph_implementation/simpleGraph.py",
#   line 152, in __from_edgelist -> raise ValueError: source column names
#   and/or destination column names not found in input. Recheck the source
#   and destination parameters
from cugraph.datasets import karate

G = karate.get_graph(download=True)

# NEW CELL

# datafile="../../data/cyber.csv"
# datafile="../../../datasets/cyber.csv"
datafile = "datasets/cyber.csv"
file = open(datafile, "rb")
Gnx = nx.read_edgelist(file)
file.close()

# NEW CELL

bc_nx_vert = nx.betweenness_centrality(Gnx)
vertex_bc = cugraph.betweenness_centrality(G)

bc_nx_edge = nx.edge_betweenness_centrality(Gnx)
edge_bc = cugraph.edge_betweenness_centrality(G)

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL

# NEW CELL
