# Copyright (c) 2020, NVIDIA CORPORATION.
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
import networkx as nx 
import cugraph
import cudf

def convert_from_nx(nxG):
    if type(nxG) == nx.classes.graph.Graph:
        G = cugraph.Graph()
    elif type(nxG) == nx.classes.digraph.DiGraph:
        G = cugraph.DiGraph()
    else:
        raise ValueError("nxG does not appear to be a NetworkX graph type")

    pdf = nx.to_pandas_edgelist(nxG)
    gdf = cudf.from_pandas(pdf)

    num_col = len(gdf.columns)
    if num_col < 2:
        raise ValueError("NetworkX graph did not contain edges")
    elif num_col == 2:
        gdf.columns = ["0", "1"]
        G.from_cudf_edgelist(gdf, "0", "1")
    else:
        gdf.columns = ["0", "1", "2"]
        G.from_cudf_edgelist(gdf, "0", "1", "2")

    del gdf
    del pdf

    return G
