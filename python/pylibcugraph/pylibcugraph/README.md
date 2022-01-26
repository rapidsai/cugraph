# `pylibcugraph`

This directory contains the sources to the `pylibcugraph` package. The sources
are primarily cython files which are built using the `setup.py` file in the
parent directory and depend on the `libcugraph_c` and `libcugraph` libraries and
headers.

## components
The `connected_components` APIs.

## structure
Internal utilities and types for use with the libcugraph C++ library.

## utilities
Utility functions.

## experimental
This subpackage defines the "experimental" APIs. many of these APIs are defined
elsewhere and simply imported into the `experimental/__init__.py` file.

## tests
pytest tests for `pylibcugraph`.
