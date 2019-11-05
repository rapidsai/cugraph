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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.community.c_spectral_clustering cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def spectralBalancedCutClustering(graph_ptr,
                                    num_clusters,
                                    num_eigen_vects=2,
                                    evs_tolerance=.00001,
                                    evs_max_iter=100,
                                    kmean_tolerance=.00001,
                                    kmean_max_iter=100):
    """
    Call cugraph::balancedCutClustering_nvgraph
    """

    cdef uintptr_t graph = graph_ptr
    cdef Graph * g = <Graph*> graph

    # Ensure that the graph has CSR adjacency list
    err = cugraph::add_adj_list(g)
    

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    df['cluster'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_cluster_col = get_gdf_column_view(df['cluster'])

    # Set the vertex identifiers
    err = g.adjList.get_vertex_identifiers(&c_identifier_col)
    

    cugraph::balancedCutClustering_nvgraph(g,
                                            num_clusters,
                                            num_eigen_vects,
                                            evs_tolerance,
                                            evs_max_iter,
                                            kmean_tolerance,
                                            kmean_max_iter,
                                            &c_cluster_col)
    

    return df

def spectralModularityMaximizationClustering(graph_ptr,
                                               num_clusters,
                                               num_eigen_vects=2,
                                               evs_tolerance=.00001,
                                               evs_max_iter=100,
                                               kmean_tolerance=.00001,
                                               kmean_max_iter=100):
    """
    Call cugraph::spectralModularityMaximization_nvgraph
    """

    cdef uintptr_t graph = graph_ptr
    cdef Graph * g = <Graph*> graph

    # Ensure that the graph has CSR adjacency list
    err = cugraph::add_adj_list(g)
    

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    df['cluster'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_cluster_col = get_gdf_column_view(df['cluster'])

    # Set the vertex identifiers
    err = g.adjList.get_vertex_identifiers(&c_identifier_col)
    

    cugraph::spectralModularityMaximization_nvgraph(g,
                                                     num_clusters,
                                                     num_eigen_vects,
                                                     evs_tolerance,
                                                     evs_max_iter,
                                                     kmean_tolerance,
                                                     kmean_max_iter,
                                                     &c_cluster_col)
    

    return df

def analyzeClustering_modularity(graph_ptr, n_clusters, clustering):
    """
    Call cugraph::analyzeClustering_modularity_nvgraph
    """
    cdef uintptr_t graph = graph_ptr
    cdef Graph * g = <Graph*> graph

    # Ensure that the graph has CSR adjacency list
    err = cugraph::add_adj_list(g)
    

    cdef gdf_column c_clustering_col = get_gdf_column_view(clustering)
    cdef float score
    cugraph::analyzeClustering_modularity_nvgraph(g, n_clusters, &c_clustering_col, &score)
    
    return score

def analyzeClustering_edge_cut(graph_ptr, n_clusters, clustering):
    """
    Call cugraph::analyzeClustering_edge_cut_nvgraph
    """
    cdef uintptr_t graph = graph_ptr
    cdef Graph * g = <Graph*> graph

    # Ensure that the graph has CSR adjacency list
    err = cugraph::add_adj_list(g)
    

    cdef gdf_column c_clustering_col = get_gdf_column_view(clustering)
    cdef float score
    cugraph::analyzeClustering_edge_cut_nvgraph(g, n_clusters, &c_clustering_col, &score)
    
    return score

def analyzeClustering_ratio_cut(graph_ptr, n_clusters, clustering):
    """
    Call cugraph::analyzeClustering_ratio_cut_nvgraph
    """
    cdef uintptr_t graph = graph_ptr
    cdef Graph * g = <Graph*> graph

    # Ensure that the graph has CSR adjacency list
    err = cugraph::add_adj_list(g)
    

    cdef gdf_column c_clustering_col = get_gdf_column_view(clustering)
    cdef float score
    cugraph::analyzeClustering_ratio_cut_nvgraph(g, n_clusters, &c_clustering_col, &score)
    
    return score
