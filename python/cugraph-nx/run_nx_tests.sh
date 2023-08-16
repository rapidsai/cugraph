#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.

# NETWORKX_GRAPH_CONVERT=cugraph is necessary to test our backend.
#
# NETWORKX_TEST_FALLBACK_TO_NX=True is optional
#   With this set, input graphs will not be converted to cugraph-nx and the networkx algorithm
#   will be called for algorithms that we don't implement or if we raise NotImplementedError.
#   This is sometimes helpful to get increased testing and coverage, but testing takes longer.
#   Without it, tests will xfail when encountering a function that we don't implement.
#
# Coverage of `cugraph_nx.algorithms` is reported and is a good sanity check that algorithms run.

# NETWORKX_GRAPH_CONVERT=cugraph NETWORKX_BACKEND_TEST_EXHAUSTIVE=True pytest --pyargs networkx "$@"
NETWORKX_TEST_BACKEND=cugraph NETWORKX_TEST_FALLBACK_TO_NX=True pytest --pyargs networkx --cov=cugraph_nx/algorithms --cov-report term-missing --no-cov-on-fail "$@"
