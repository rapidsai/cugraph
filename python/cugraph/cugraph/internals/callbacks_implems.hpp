/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <Python.h>
#include <cugraph/legacy/internals.hpp>

#include <iostream>

namespace cugraph {
namespace internals {

class DefaultGraphBasedDimRedCallback : public GraphBasedDimRedCallback {
 public:
  PyObject* get_numba_matrix(void* positions)
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;

    if (isFloat) {
      return PyObject_CallMethod(
        pycl, "get_numba_matrix", "(l(ll)s)", positions, n, n_components, "float32");
    } else {
      return PyObject_CallMethod(
        pycl, "get_numba_matrix", "(l(ll)s)", positions, n, n_components, "float64");
    }
  }

  void on_preprocess_end(void* positions) override
  {
    PyObject* numba_matrix = get_numba_matrix(positions);
    PyObject* res =
      PyObject_CallMethod(this->pyCallbackClass, "on_preprocess_end", "(O)", numba_matrix);
    Py_DECREF(numba_matrix);
    Py_DECREF(res);
  }

  void on_epoch_end(void* positions) override
  {
    PyObject* numba_matrix = get_numba_matrix(positions);
    PyObject* res = PyObject_CallMethod(this->pyCallbackClass, "on_epoch_end", "(O)", numba_matrix);
    Py_DECREF(numba_matrix);
    Py_DECREF(res);
  }

  void on_train_end(void* positions) override
  {
    PyObject* numba_matrix = get_numba_matrix(positions);
    PyObject* res = PyObject_CallMethod(this->pyCallbackClass, "on_train_end", "(O)", numba_matrix);
    Py_DECREF(numba_matrix);
    Py_DECREF(res);
  }

 public:
  PyObject* pyCallbackClass;
};

}  // namespace internals
}  // namespace cugraph
