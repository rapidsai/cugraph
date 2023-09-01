#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# NETWORKX_GRAPH_CONVERT=cugraph
#   Used by networkx versions 3.0 and 3.1
#   Must be set to "cugraph" to test the cugraph-nx backend.
#
# NETWORKX_TEST_BACKEND=cugraph
#   Replaces NETWORKX_GRAPH_CONVERT for networkx versions >=3.2
#   Must be set to "cugraph" to test the cugraph-nx backend.
#
# NETWORKX_FALLBACK_TO_NX=True (optional)
#   Used by networkx versions >=3.2.  With this set, input graphs will not be
#   converted to cugraph-nx and the networkx algorithm will be called for
#   algorithms that we don't implement or if we raise NotImplementedError.
#   This is sometimes helpful to get increased testing and coverage, but
#   testing takes longer.  Without it, tests will xfail when encountering a
#   function that we don't implement.
#
# Coverage of `cugraph_nx.algorithms` is reported and is a good sanity check
# that algorithms run.

# Warning: cugraph has a .coveragerc file in the <repo root>/python directory,
# so be mindful of its contents and the CWD when running.
# FIXME: should something be added to detect/prevent the above?

NETWORKX_GRAPH_CONVERT=cugraph \
NETWORKX_TEST_BACKEND=cugraph \
NETWORKX_FALLBACK_TO_NX=True \
    pytest \
    --pyargs networkx \
    --cov=cugraph_nx.algorithms \
    --cov-report term-missing \
    --no-cov-on-fail \
    "$@"
