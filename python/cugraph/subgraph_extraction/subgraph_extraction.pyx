# Copyright (c) 2019, NVIDIA CORPORATION.
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

from c_nvgraph cimport * 
from c_graph cimport * 
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP
import cudf
from librmm_cffi import librmm as rmm
import numpy as np

cpdef subgraph(G, vertices):
    """
    Compute a subgraph of the existing graph including only the specified 
    vertices.  This algorithm works for both directed and undirected graphs,
    it does not actually traverse the edges, simply pulls out any edges that
    are incident on vertices that are both contained in the vertices list.
    
    Parameters
    ----------
    G : cuGraph.Graph                  
       cuGraph graph descriptor
    vertices : cudf.Series
        Specifies the vertices of the induced subgraph
    
    Returns
    -------
    Sg : cuGraph.Graph
        A graph object containing the subgraph induced by the given vertex set.
        
    Example:
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> verts = numpy.zeros(3, dtype=np.int32)
    >>> verts[0] = 0
    >>> verts[1] = 1
    >>> verts[2] = 2
    >>> sverts = cudf.Series(verts)
    >>> Sg = cuGraph.subgraph(G, sverts)
    """

    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph

    resultGraph = Graph()
    cdef uintptr_t rGraph = resultGraph.graph_ptr
    cdef gdf_graph* rg = <gdf_graph*>rGraph

    cdef gdf_column vert_col = get_gdf_column_view(vertices)

    err = gdf_extract_subgraph_vertex_nvgraph(g, &vert_col, rg)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return resultGraph
