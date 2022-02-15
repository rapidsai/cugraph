# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

from cugraph.community.spectral_clustering cimport balancedCutClustering as c_balanced_cut_clustering
from cugraph.community.spectral_clustering cimport spectralModularityMaximization as c_spectral_modularity_maximization
from cugraph.community.spectral_clustering cimport analyzeClustering_modularity as c_analyze_clustering_modularity
from cugraph.community.spectral_clustering cimport analyzeClustering_edge_cut as c_analyze_clustering_edge_cut
from cugraph.community.spectral_clustering cimport analyzeClustering_ratio_cut as c_analyze_clustering_ratio_cut
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper
from libc.stdint cimport uintptr_t
import cugraph
import cudf
import numpy as np


def spectralBalancedCutClustering(input_graph,
                                  num_clusters,
                                  num_eigen_vects=2,
                                  evs_tolerance=.00001,
                                  evs_max_iter=100,
                                  kmean_tolerance=.00001,
                                  kmean_max_iter=100):
    """
    Call balancedCutClustering_nvgraph
    """
    if isinstance(input_graph, cugraph.Graph):
        if input_graph.is_directed():
            raise ValueError("directed graphs are not supported")
    else:
        raise TypeError(f"only cugraph.Graph objects are supported, got: {type(input_graph)}")
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    weights = None

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(np.full(num_edges, 1.0, dtype=np.float32))

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['cluster'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_cluster = df['cluster'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                              <float*>c_weights, num_verts, num_edges)

        graph_float.get_vertex_identifiers(<int*>c_identifier)
        c_balanced_cut_clustering(graph_float,
                                  num_clusters,
                                  num_eigen_vects,
                                  evs_tolerance,
                                  evs_max_iter,
                                  kmean_tolerance,
                                  kmean_max_iter,
                                  <int*>c_cluster)
    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                <double*>c_weights, num_verts, num_edges)

        graph_double.get_vertex_identifiers(<int*>c_identifier)
        c_balanced_cut_clustering(graph_double,
                                  num_clusters,
                                  num_eigen_vects,
                                  evs_tolerance,
                                  evs_max_iter,
                                  kmean_tolerance,
                                  kmean_max_iter,
                                  <int*>c_cluster)

    return df

def spectralModularityMaximizationClustering(input_graph,
                                             num_clusters,
                                             num_eigen_vects=2,
                                             evs_tolerance=.00001,
                                             evs_max_iter=100,
                                             kmean_tolerance=.00001,
                                             kmean_max_iter=100):
    """
    Call spectralModularityMaximization_nvgraph
    """
    if isinstance(input_graph, cugraph.Graph):
        if input_graph.is_directed():
            raise ValueError("directed graphs are not supported")
    else:
        raise TypeError(f"only cugraph.Graph objects are supported, got: {type(input_graph)}")
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    if input_graph.adjlist.weights is None:
        raise Exception("spectral modularity maximization must be called on a graph with weights")

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['cluster'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_cluster = df['cluster'].__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                              <float*>c_weights, num_verts, num_edges)

        graph_float.get_vertex_identifiers(<int*>c_identifier)
        c_spectral_modularity_maximization(graph_float,
                                           num_clusters,
                                           num_eigen_vects,
                                           evs_tolerance,
                                           evs_max_iter,
                                           kmean_tolerance,
                                           kmean_max_iter,
                                           <int*>c_cluster)
    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                <double*>c_weights, num_verts, num_edges)

        graph_double.get_vertex_identifiers(<int*>c_identifier)
        c_spectral_modularity_maximization(graph_double,
                                           num_clusters,
                                           num_eigen_vects,
                                           evs_tolerance,
                                           evs_max_iter,
                                           kmean_tolerance,
                                           kmean_max_iter,
                                           <int*>c_cluster)

    return df

def analyzeClustering_modularity(input_graph, n_clusters, clustering):
    """
    Call analyzeClustering_modularity_nvgraph
    """
    if isinstance(input_graph, cugraph.Graph):
        if input_graph.is_directed():
            raise ValueError("directed graphs are not supported")
    else:
        raise TypeError(f"only cugraph.Graph objects are supported, got: {type(input_graph)}")
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])

    score = None
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is None:
        raise Exception("analyze clustering modularity must be called on a graph with weights")
    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(np.full(num_edges, 1.0, dtype=np.float32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_cluster = clustering.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double
    cdef float score_float
    cdef double score_double

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                              <float*>c_weights, num_verts, num_edges)

        c_analyze_clustering_modularity(graph_float,
                                        n_clusters,
                                        <int*> c_cluster,
                                        &score_float)

        score = score_float
    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                <double*>c_weights, num_verts, num_edges)

        c_analyze_clustering_modularity(graph_double,
                                        n_clusters,
                                        <int*> c_cluster,
                                        &score_double)
        score = score_double

    return score

def analyzeClustering_edge_cut(input_graph, n_clusters, clustering):
    """
    Call analyzeClustering_edge_cut_nvgraph
    """
    if isinstance(input_graph, cugraph.Graph):
        if input_graph.is_directed():
            raise ValueError("directed graphs are not supported")
    else:
        raise TypeError(f"only cugraph.Graph objects are supported, got: {type(input_graph)}")
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    score = None
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(np.full(num_edges, 1.0, dtype=np.float32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_cluster = clustering.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double
    cdef float score_float
    cdef double score_double

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                              <float*>c_weights, num_verts, num_edges)

        c_analyze_clustering_edge_cut(graph_float,
                                      n_clusters,
                                      <int*> c_cluster,
                                      &score_float)

        score = score_float
    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                <double*>c_weights, num_verts, num_edges)

        c_analyze_clustering_edge_cut(graph_double,
                                      n_clusters,
                                      <int*> c_cluster,
                                      &score_double)
        score = score_double

    return score

def analyzeClustering_ratio_cut(input_graph, n_clusters, clustering):
    """
    Call analyzeClustering_ratio_cut_nvgraph
    """
    if isinstance(input_graph, cugraph.Graph):
        if input_graph.is_directed():
            raise ValueError("directed graphs are not supported")
    else:
        raise TypeError(f"only cugraph.Graph objects are supported, got: {type(input_graph)}")
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    score = None
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(np.full(num_edges, 1.0, dtype=np.float32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_cluster = clustering.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double
    cdef float score_float
    cdef double score_double

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                              <float*>c_weights, num_verts, num_edges)

        c_analyze_clustering_ratio_cut(graph_float,
                                       n_clusters,
                                       <int*> c_cluster,
                                       &score_float)

        score = score_float
    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                <double*>c_weights, num_verts, num_edges)

        c_analyze_clustering_ratio_cut(graph_double,
                                       n_clusters,
                                       <int*> c_cluster,
                                       &score_double)
        score = score_double

    return score
