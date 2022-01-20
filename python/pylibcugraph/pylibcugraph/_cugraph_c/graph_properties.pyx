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

from pylibcugraph._cugraph_c.cugraph_api cimport (
     bool_t,
)


cdef class EXPERIMENTAL__GraphProperties:
    """
    """
    def __cinit__(self):
        self.c_graph_properties.is_symmetric = bool_t.FALSE
        self.c_graph_properties.is_multigraph = bool_t.FALSE

    @property
    def is_symmetric(self):
        return bool(self.c_graph_properties.is_symmetric)

    @is_symmetric.setter
    def is_symmetric(self, value):
        if not(isinstance(value, (int, bool))):
            raise TypeError(f"expected int or bool, got {type(value)}")

        self.c_graph_properties.is_symmetric = int(value)

    @property
    def is_multigraph(self):
        return bool(self.c_graph_properties.is_multigraph)

    @is_multigraph.setter
    def is_multigraph(self, value):
        if not(isinstance(value, (int, bool))):
            raise TypeError(f"expected int or bool, got {type(value)}")

        self.c_graph_properties.is_multigraph = int(value)
