# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t


cdef extern from "cugraph.h":
    cdef int get_device(void *ptr)


def device_of_gpu_pointer(g):
    cdef uintptr_t cptr = g.device_ctypes_pointer.value
    return get_device(<void*> cptr)

