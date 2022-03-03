#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
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
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# import raft
from libcpp.memory cimport shared_ptr
from rmm._lib.cuda_stream_view cimport cuda_stream_per_thread
from rmm._lib.cuda_stream_view cimport cuda_stream_view

from .cuda cimport Stream
from .cuda import CudaRuntimeError


cdef class Handle:
    """
    Handle is a lightweight python wrapper around the corresponding C++ class
    of handle_t exposed by RAFT's C++ interface. Refer to the header file
    raft/handle.hpp for interface level details of this struct

    Examples
    --------

    .. code-block:: python

        from raft.common import Stream, Handle
        stream = Stream()
        handle = Handle(stream)

        # call algos here

        # final sync of all work launched in the stream of this handle
        # this is same as `raft.cuda.Stream.sync()` call, but safer in case
        # the default stream inside the `handle_t` is being used
        handle.sync()
        del handle  # optional!
    """

    def __cinit__(self, stream: Stream = None, n_streams=0):
        self.n_streams = n_streams
        if n_streams > 0:
            self.stream_pool.reset(new cuda_stream_pool(n_streams))

        cdef cuda_stream_view c_stream
        if stream is None:
            # this constructor will construct a "main" handle on
            # per-thread default stream, which is non-blocking
            self.c_obj.reset(new handle_t(cuda_stream_per_thread,
                                          self.stream_pool))
        else:
            # this constructor constructs a handle on user stream
            c_stream = cuda_stream_view(stream.getStream())
            self.c_obj.reset(new handle_t(c_stream,
                                          self.stream_pool))

    def sync(self):
        """
        Issues a sync on the stream set for this handle.
        """
        self.c_obj.get()[0].sync_stream()

    def getHandle(self):
        return <size_t> self.c_obj.get()

    def __getstate__(self):
        return self.n_streams

    def __setstate__(self, state):
        self.n_streams = state
        if self.n_streams > 0:
            self.stream_pool.reset(new cuda_stream_pool(self.n_streams))

        self.c_obj.reset(new handle_t(cuda_stream_per_thread,
                                      self.stream_pool))
