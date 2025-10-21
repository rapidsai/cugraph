# SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libc.stdint cimport uintptr_t
from numba.cuda.api import from_cuda_array_interface
import numpy as np


cdef extern from "Python.h":
    cdef cppclass PyObject


cdef extern from "callbacks_implems.hpp" namespace "cugraph::internals":
    cdef cppclass Callback:
        pass

    cdef cppclass DefaultGraphBasedDimRedCallback(Callback):
        void setup(int n, int d) except +
        void on_preprocess_end(void *positions) except +
        void on_epoch_end(void *positions) except +
        void on_train_end(void *positions) except +
        PyObject* pyCallbackClass


cdef class PyCallback:

    def get_numba_matrix(self, positions, shape, typestr):

        sizeofType = 4 if typestr == "float32" else 8
        desc = {
            'shape': shape,
            'strides': (sizeofType, shape[0]*sizeofType),
            'typestr': typestr,
            'data': [positions],
            'order': 'C',
            'version': 1
        }

        return from_cuda_array_interface(desc)


cdef class GraphBasedDimRedCallback(PyCallback):
    """
    Usage
    -----

    class CustomCallback(GraphBasedDimRedCallback):
        def on_preprocess_end(self, positions):
            print(positions.copy_to_host())

        def on_epoch_end(self, positions):
            print(positions.copy_to_host())

        def on_train_end(self, positions):
            print(positions.copy_to_host())
    """

    cdef DefaultGraphBasedDimRedCallback native_callback

    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
