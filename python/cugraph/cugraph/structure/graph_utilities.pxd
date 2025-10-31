# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
from libcpp.vector cimport vector

from rmm.librmm.device_buffer cimport device_buffer

from pylibraft.common.handle cimport handle_t


cdef extern from "<utility>" namespace "std" nogil:
    cdef device_buffer move(device_buffer)
    cdef unique_ptr[device_buffer] move(unique_ptr[device_buffer])
