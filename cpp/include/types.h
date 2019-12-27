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

// TODO : [WIP] improve graph class and types 

namespace cugraph {

template <typename T>
void ALLOC_FREE_TRY(T* ptr);


typedef enum prop_type{PROP_UNDEF, PROP_FALSE, PROP_TRUE} PropType;

struct Graph_properties {
  bool directed;
  bool weighted;
  bool multigraph;
  bool bipartite;
  bool tree;
  PropType has_negative_edges;
  Graph_properties() : directed(false), weighted(false), multigraph(false), bipartite(false), tree(false), has_negative_edges(PROP_UNDEF){}
};

template <typename VT, typename WT>
struct edge_list{
  VT *src_indices; // rowInd
  VT *dest_indices; // colInd
  WT *edge_data; //val
  int ownership = 0; // 0 if all columns were provided by the user, 1 if cugraph crated everything, other values can be use for other cases
  edge_list() : src_indices(nullptr), dest_indices(nullptr), edge_data(nullptr){}
  ~edge_list() {
    if (ownership == 1 ) {
      ALLOC_FREE_TRY(src_indices);
      ALLOC_FREE_TRY(dest_indices);
      ALLOC_FREE_TRY(edge_data);
    }
    else if (ownership == 2 )
    {
      ALLOC_FREE_TRY(src_indices);
    }
  }
};

template <typename VT, typename WT>
struct adj_list{
  VT *offsets; // rowPtr
  VT *indices; // colInd
  WT *edge_data; //val
  int ownership = 0; // 0 if all columns were provided by the user, 1 if cugraph crated everything, other values can be use for other cases
  adj_list() : offsets(nullptr), indices(nullptr), edge_data(nullptr){}
  ~adj_list() {
    if (ownership == 1 ) {
      ALLOC_FREE_TRY(offsets);
      ALLOC_FREE_TRY(indices);
      ALLOC_FREE_TRY(edge_data);
    }
  }
  void get_vertex_identifiers(VT *identifiers);
  void get_source_indices(VT *indices);
};

struct dynamic{
  void *data; // handle to the dynamic graph struct
};

struct Graph{
    size_t v, e;
    edge_list *edgeList; // COO
    adj_list *adjList; //CSR
    adj_list *transposedAdjList; //CSC
    dynamic *dynAdjList; //dynamic 
    Graph_properties *prop;
    size_type numberOfVertices;
    Graph() : v(0), e(0), edgeList(nullptr), adjList(nullptr), transposedAdjList(nullptr), dynAdjList(nullptr), prop(nullptr), numberOfVertices(0) {}
    ~Graph() {
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

} //namespace cugraph