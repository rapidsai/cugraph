# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cugraph.structure.graph_primtypes cimport *


cdef extern from "cugraph/algorithms.hpp" namespace "cugraph::ext_raft":

    cdef void balancedCutClustering[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        const int num_clusters,
        const int num_eigen_vects,
        const float evs_tolerance,
        const int evs_max_iter,
        const float kmean_tolerance,
        const int kmean_max_iter,
        VT* clustering) except +

    cdef void spectralModularityMaximization[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        const int n_clusters,
        const int n_eig_vects,
        const float evs_tolerance,
        const int evs_max_iter,
        const float kmean_tolerance,
        const int kmean_max_iter,
        VT* clustering) except +

    cdef void analyzeClustering_modularity[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        const int n_clusters,
        const VT* clustering,
        WT* score) except +

    cdef void analyzeClustering_edge_cut[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        const int n_clusters,
        const VT* clustering,
        WT* score) except +

    cdef void analyzeClustering_ratio_cut[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        const int n_clusters,
        const VT* clustering,
        WT* score) except +
