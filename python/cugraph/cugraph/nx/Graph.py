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

from distutils.command.config import dump_file
import pandas as pd
import cudf
import cugraph

from cugraph.experimental import PropertyGraph

class Graph():
    """
    Class which replaces networkX graph allowing graph building on top of Property_Graph
    taking advantage of cuda DataFrames instead of building on host memory then doing
    transfers to gpu often/later.
    """

    # default name for the vertex column in the dataframe when none is provided
    node_name_column = "_VERTEX_"

    # default name for the source or first edge column in the dataframe when 
    # none is provided
    src_col_name = "_SRC_"
    
    # default name for the destination or second edge column in the dataframe when 
    # none is provided
    dst_col_name = "_DST_"

    # defines the type of the nodes inserted in the property graph
    # initially will be str or int
    node_data_type = None
 

    def __init__(self):
        self.__pG = PropertyGraph()


    def pG(self):
        return self.__pG


    # checks the node name to avoid mixed types
    # sets the type if it is the first insert in the graph
    def check_type(self,data):
        """
        Checks the node name to avoid mixed types
        sets the type if it is the first insert in the graph 

        Parameters
        ----------
            data : single item to represent the node id in the graph
            Must be able to be represented as an str.
        """
        if (self.node_data_type == None):
            self.node_data_type = (type(data))
        if (type(data) != self.node_data_type):
            raise TypeError("type_name must be a string, got: "
                            f"{type(data)}")           
   

    # Add a single node with or without additional properties
    # FIXME handle input of a dataframe with nodes
    def add_node(self,node):
        property_column_names = None
        columns=[self.node_name_column]
        ndata = None
        df_data = list()
        if isinstance(node,tuple):
            property_columns = list()
            self.check_type(node[0])
            ndata = [(node[0])]
            if isinstance(node[1], dict):
                for property in node[1].keys():
                    property_columns.append(property)
                    columns.append(property)
                    ndata.append(node[1].get(property))
            df_data.append(ndata)
        else:
            self.check_type(node)
            df_data = [[(node)]]
        node_df = cudf.DataFrame(data=df_data,columns=columns)
        self.__pG.add_vertex_data(node_df,
            type_name=self.node_name_column,
            vertex_id_column=self.node_name_column,
            property_columns=property_column_names
            )


    def add_nodes_from(self,nodes):
        """
        Adds nodes from a provided list.

        Parameters
        ----------
        nodes : must be a list of nodes with or without properties in the 
            form of a dictionary.
        """
        if not isinstance(nodes,list):
            raise TypeError("type_name must be a list, got: "
                            f"{type(nodes)}")
        for node in nodes:
            # print(f'Handling node {node}')
            self.add_node(node)
           

    def number_of_nodes(self):
        """
        Counts the nodes(vertices) in the graph

        Returns
        -------
        int with the number of nodes in the underlying Property_Graph 
        """
        return self.__pG.num_vertices

    def remove_node(self, node):
        """
        Checks the node name to avoid mixed types
        sets the type if it is the first insert in the graph 

        Parameters
        ----------
        data : single item to represent the node id in the graph.
            Must be able to be represented as an str.
        """

        print("Not yet implemented")
        return


    def remove_nodes_from(self):
        print("Not yet implemented")
        return


    def order(self):
        print("Not yet implemented")
        return


    def has_node(self):
        print("Not yet implemented")
        return


    def add_weighted_edges_from(self):
        print("Not yet implemented")
        return


    def remove_edge(self):
        print("Not yet implemented")
        return


    def remove_edges_from(self):
        print("Not yet implemented")
        return


    def update(self):
        print("Not yet implemented")
        return


    def has_edge(self):
        print("Not yet implemented")
        return

    def get_connections(self, nodelist):
        edgefilter = f"{self.__pG.src_col_name}.isin({nodelist}) | {self.__pG.src_col_name}.isin({nodelist})"
        selected_edges = self.__pG.select_edges(edgefilter)
        subgraph = self.__pG.extract_subgraph(create_using=cugraph.Graph(directed=True),
                 selection=selected_edges,
                 default_edge_weight=1.0,
                 allow_multi_edges=True)
        return subgraph

    def neighbors(self):
        print("Not yet implemented")
        return


    def get_edge_data(self):
        print("Not yet implemented")
        return


    def adjacency(self):
        print("Not yet implemented")
        return


    def degree(self):    
        print("Not yet implemented")
        return


    def clear(self):
        print("Not yet implemented")
        return


    def clear_edges(self):
        print("Not yet implemented")
        return


    def read_edgelist(self):
        print("Not yet implemented")
        return


    def read_weighted_edgelist(self):
        print("Not yet implemented")
        return


    def read_adjlist(self):
        print("Not yet implemented")
        return


    def number_of_edges(self):
        return self.__pG.num_edges


    def add_edge(self, u,v):
        self.check_type(u)
        self.check_type(v)
        ndata = [[u,v]]
        edge_df = cudf.DataFrame(columns=None,data=ndata)
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
                print("Reached with no list or column titles")
                in_data = edges
                column_names=[self.src_col_name,self.dst_col_name]
                # print(f'Column names = {column_names}')
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


    def as_cugraph(self):
        return self.__pG.extract_subgraph()


    def number_of_edges(self):
        return self.__pG.num_edges


    def nodes(self):
        return self.__pG.get_vertices().values


    def edges(self):
       return self.__pG.select_edges()
