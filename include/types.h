#pragma once 

void gdf_col_delete(gdf_column* col);

void gdf_col_release(gdf_column* col);

struct gdf_graph_properties {
  bool directed;
  bool weighted;
  bool multigraph;
  bool bipartite;
  bool tree;
  gdf_graph_properties() : directed(false), weighted(false), multigraph(false), bipartite(false), tree(false){}
};

struct gdf_edge_list{
  gdf_column *src_indices; // rowInd
  gdf_column *dest_indices; // colInd
  gdf_column *edge_data; //val
  int ownership = 0; // 0 if all columns were provided by the user, 1 if cugraph crated everything, other values can be use for other cases
  gdf_edge_list() : src_indices(nullptr), dest_indices(nullptr), edge_data(nullptr){}
  ~gdf_edge_list() {
    if (ownership == 0 ) {
      gdf_col_release(src_indices);
      gdf_col_release(dest_indices);
      gdf_col_release(edge_data);
    }
    //else if (ownership == 2 )
    //{
    //  gdf_col_release(src_indices);
    //  gdf_col_release(dest_indices);
    //  gdf_col_delete(edge_data);
    //}
    else {
      gdf_col_delete(src_indices);
      gdf_col_delete(dest_indices);
      gdf_col_delete(edge_data);
    }
  }
};

struct gdf_adj_list{
  gdf_column *offsets; // rowPtr
  gdf_column *indices; // colInd
  gdf_column *edge_data; //val
  int ownership = 0; // 0 if all columns were provided by the user, 1 if cugraph crated everything, other values can be use for other cases
  gdf_adj_list() : offsets(nullptr), indices(nullptr), edge_data(nullptr){}
  ~gdf_adj_list() {
    if (ownership == 0 ) {
      gdf_col_release(offsets);
      gdf_col_release(indices);
      gdf_col_release(edge_data);
    }
    //else if (ownership == 2 )
    //{
    //  gdf_col_release(offsets);
    //  gdf_col_release(indices);
    //  gdf_col_delete(edge_data);
    //}
    else {
      gdf_col_delete(offsets);
      gdf_col_delete(indices);
      gdf_col_delete(edge_data);
    }
  }
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
  gdf_graph() : edgeList(nullptr), adjList(nullptr), transposedAdjList(nullptr), dynAdjList(nullptr), prop(nullptr) {}
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
