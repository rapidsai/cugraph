# cugraph_python

Nanobind bindings for a subset of the cuGraph C++ API.

This package is a prototype. It builds graphs from an owning COO and runs single-GPU BFS.
It depends on the `libcugraph` wheel for the C++ library and does not use the C API.

Build it the same way as the other Python packages in this repository:

```bash
./build.sh cugraph_python
```
