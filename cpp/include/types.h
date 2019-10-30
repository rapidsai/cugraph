/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

void gdf_col_delete(gdf_column* col);

void gdf_col_release(gdf_column* col);

typedef enum gdf_prop_type{GDF_PROP_UNDEF, GDF_PROP_FALSE, GDF_PROP_TRUE} GDFPropType;

struct gdf_graph_properties {
  bool directed;
  bool weighted;
  bool multigraph;
  bool bipartite;
  bool tree;
  GDFPropType has_negative_edges;
  gdf_graph_properties() : directed(false), weighted(false), multigraph(false), bipartite(false), tree(false), has_negative_edges(GDF_PROP_UNDEF){}
};

struct gdf_edge_list{
  gdf_column *src_indices; // rowInd
  gdf_column *dest_indices; // colInd
  gdf_column *edge_data; //val
  gdf_edge_list() : src_indices(nullptr), dest_indices(nullptr), edge_data(nullptr){}
  ~gdf_edge_list() {
    gdf_col_release(src_indices);
    gdf_col_release(dest_indices);
    gdf_col_release(edge_data);
  }
};

struct gdf_adj_list{
  gdf_column *offsets; // rowPtr
  gdf_column *indices; // colInd
  gdf_column *edge_data; //val
  gdf_adj_list() : offsets(nullptr), indices(nullptr), edge_data(nullptr){}
  ~gdf_adj_list() {
    gdf_col_release(offsets);
    gdf_col_release(indices);
    gdf_col_release(edge_data);
  }
  gdf_error get_vertex_identifiers(gdf_column *identifiers);
  gdf_error get_source_indices(gdf_column *indices);

};

struct gdf_dynamic{
  void *data; // handle to the dynamic graph struct
};

struct gdf_graph{
  gdf_edge_list *edgeList; // COO
  gdf_adj_list *adjList; //CSR
  gdf_adj_list *transposedAdjList; //CSC
  gdf_dynamic *dynAdjList; //dynamic 
  gdf_graph_properties *prop;
  gdf_size_type numberOfVertices;
  gdf_graph() : edgeList(nullptr), adjList(nullptr), transposedAdjList(nullptr), dynAdjList(nullptr), prop(nullptr), numberOfVertices(0) {}
  ~gdf_graph() {
    if (edgeList) 
        delete edgeList;
    if (adjList) 
        delete adjList;
    if (transposedAdjList) 
        delete transposedAdjList;
    if (dynAdjList) 
        delete dynAdjList;
    if (prop) 
        delete prop;
  }
};
