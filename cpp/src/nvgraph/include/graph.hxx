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

#include <cstdlib>
#include <cstddef> // size_t
#include <iostream> 

#include <graph_visitors.hxx>//
//
namespace nvgraph
{

#define DEFINE_VISITABLE(T) \
virtual void Accept(VisitorBase& guest) \
{ BaseVisitableGraph<T>::AcceptImpl(*this, guest); }

template<typename T>
struct BaseVisitableGraph
{
  virtual void Accept(VisitorBase& v) = 0;

  virtual ~BaseVisitableGraph(void)
  {
  }
protected:
  template<typename Host>
  static void AcceptImpl(Host& visited, VisitorBase& guest)
  {
	if( Visitor<Host>* p = dynamic_cast<Visitor<Host>*>(&guest))
	  {
		p->Visit(visited);
	  }
  }
};

template<typename IndexType_>
class Graph: public BaseVisitableGraph<IndexType_>
{
public:
    typedef IndexType_ IndexType;
    
protected:
    size_t num_vertices;
    size_t num_edges;
    Graph<IndexType> *parent;
    Graph<IndexType> *child;

public:
    /*! Construct an empty \p Graph.
     */
    Graph()
        : num_vertices(0),num_edges(0) {}

    /*! Construct a \p Graph with a specific number of vertices.
     *
     *  \param vertices Number of vertices.
     */
    Graph(size_t vertices)
        : num_vertices(vertices), num_edges(0) {}

    /*! Construct a \p Graph with a specific number of vertices and edges.
     *
     *  \param vertices Number of vertices.
     *  \param edges Number of edges.
     */
    Graph(size_t vertices, size_t edges)
        : num_vertices(vertices), num_edges(edges) {}

    /*! Construct a \p CsrGraph from another graph.
     *
     *  \param CsrGraph Another graph in csr
     */
    Graph(const Graph& gr)
    {
        num_vertices = gr.get_num_vertices();
        num_edges = gr.get_num_edges();
    }

    inline void set_num_vertices(IndexType_ p_num_vertices) { num_vertices = p_num_vertices; }
    inline void set_num_edges(IndexType_ p_num_edges) { num_edges = p_num_edges; }
    inline size_t get_num_vertices() const { return num_vertices; }
    inline size_t get_num_edges() const { return num_edges; }
    /*! Resize graph dimensions
     *
     *  \param num_rows Number of vertices.
     *  \param num_cols Number of edges.
     */
   //inline void resize(size_t vertices, size_t edges)
   //{
   //    num_vertices = vertices;
   //    num_edges = edges;
   //}

    //Accept method injection
    DEFINE_VISITABLE(IndexType_)
};

} // end namespace nvgraph

