# Copyright (c) 2022, NVIDIA CORPORATION.
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

# Have cython use python 3 syntax
# cython: language_level = 3

cdef class GraphProperties:
    """
    Class wrapper around C cugraph_graph_properties_t struct
    """
    def __cinit__(self, is_symmetric=False, is_multigraph=False):
        self.c_graph_properties.is_symmetric = is_symmetric
        self.c_graph_properties.is_multigraph = is_multigraph

    # Pickle support methods: get args for __new__ (__cinit__), get/set state
    def __getnewargs_ex__(self):
        is_symmetric = self.c_graph_properties.is_symmetric
        is_multigraph = self.c_graph_properties.is_multigraph
        return ((),{"is_symmetric":is_symmetric, "is_multigraph":is_multigraph})

    def __getstate__(self):
        return ()

    def __setstate__(self, state):
        pass

    @property
    def is_symmetric(self):
        return bool(self.c_graph_properties.is_symmetric)

    @is_symmetric.setter
    def is_symmetric(self, value):
        self.c_graph_properties.is_symmetric = value

    @property
    def is_multigraph(self):
        return bool(self.c_graph_properties.is_multigraph)

    @is_multigraph.setter
    def is_multigraph(self, value):
        self.c_graph_properties.is_multigraph = value
