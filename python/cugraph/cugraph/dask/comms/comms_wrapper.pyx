# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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


from pylibraft.common.handle cimport *
from cugraph.dask.comms.comms cimport init_subcomms as c_init_subcomms


def init_subcomms(handle, row_comm_size):
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t
    c_init_subcomms(handle_[0], row_comm_size)
