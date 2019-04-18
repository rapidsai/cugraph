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

#ifndef GRAPH_VISITORS_HXX
#define GRAPH_VISITORS_HXX

namespace nvgraph
{
  //PROBLEM: using Visitor Design Pattern over a 
  //         hierarchy of visitees that depend on 
  //         different number of template arguments
  //
  //SOLUTION:use Acyclic Visitor
  //         (A. Alexandrescu, "Modern C++ Design", Section 10.4), 
  //         where *concrete* Visitors must be parameterized by all 
  //         the possibile template args of the Visited classes (visitees);
  //
  struct VisitorBase
  {
    virtual ~VisitorBase(void)
    {
    }
  };

  template<typename T>
  struct Visitor
  {
    virtual void Visit(T& ) = 0;
    virtual ~Visitor() { }
  };
}//end namespace
#endif

