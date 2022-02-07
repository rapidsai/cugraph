# Copyright (c) 2022 NVIDIA CORPORATION.
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
# limitations under the License.import networkx as nx

import pandas as pd
import cudf
import cugraph

from cugraph.experimental import PropertyGraph

class Graph():

    def __init__(self):
        self.__pG = PropertyGraph()


    def pG(self):
        return self.__pG


    def add_node(self,node):
        ndata = [[node]]
        df = pd.DataFrame(data=ndata,columns=None)
        node_df = cudf.from_pandas(df)
        self.__pG.add_vertex_data(node_df,df.columns[0],type_name="")
        

    def number_of_nodes(self):
        return self.__pG.num_vertices


    def add_edge(self, u,v):
        ndata = [[u,v]]
        edge_df = cudf.DataFrame(columns=None,data=ndata)
        print(ndata)
        self.__pG.add_edge_data(edge_df,
                                vertex_id_columns=(edge_df.columns[0], edge_df.columns[1]),
                                type_name="")
       

    def add_edges_from(self, edges):
        # is not already a dataframe so make one
        if not isinstance(edges,cudf.DataFrame):
            props = None
            in_data = None
            column_names=None
            if  not isinstance(edges[0],list):
                in_data = edges      
            if  isinstance(edges[0],list) and len(list(edges))  == 2:
                column_names = edges[0]
                in_data = edges[1]
            if isinstance(edges[0],list) and len(list(edges)) > 2:
                print(f"Handle attribute data size={len(list(edges))}")
                column_names = edges[0]
                in_data = edges[1]
                props = [edges[2]]
            df = cudf.DataFrame(data=in_data,columns=column_names)
        else:
            df = edges
        self.__pG.add_edge_data(df,
                                vertex_id_columns=(df.columns[0], df.columns[1]),
                                type_name="")


    @property
    def as_cugraph(self):
        return self.__pG.extract_subgraph()


    def number_of_edges(self):
        return self.__pG.num_edges


    def edges(self,nBunch=None):
        return self.__nG.edges()
