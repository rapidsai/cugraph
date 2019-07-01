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

from cugraph.structure.c_graph cimport *


cdef extern from "cugraph.h":
    cdef gdf_error gdf_overlap (gdf_graph * graph,
                                gdf_column * weights,
                                gdf_column * result)
    
    cdef gdf_error gdf_overlap_list(gdf_graph * graph,
                                    gdf_column * weights,
                                    gdf_column * first,
                                    gdf_column * second,
                                    gdf_column * result)
