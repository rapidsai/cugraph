# SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t


'''
cdef extern from "cugraph.h" namespace "cugraph":
    cdef int get_device(void *ptr)


def device_of_gpu_pointer(g):
    cdef uintptr_t cptr = g.device_ctypes_pointer.value
    return get_device(<void*> cptr)
'''
