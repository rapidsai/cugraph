#!/usr/bin/env bash
#
# Copyright (c) 2023-2024, NVIDIA CORPORATION.
#
# NETWORKX_GRAPH_CONVERT=cugraph
#   Used by networkx versions 3.0 and 3.1
#   Must be set to "cugraph" to test the nx-cugraph backend.
#
# NETWORKX_TEST_BACKEND=cugraph
#   Replaces NETWORKX_GRAPH_CONVERT for networkx versions >=3.2
#   Must be set to "cugraph" to test the nx-cugraph backend.
#
# NETWORKX_FALLBACK_TO_NX=True (optional)
#   Used by networkx versions >=3.2.  With this set, input graphs will not be
#   converted to nx-cugraph and the networkx algorithm will be called for
#   algorithms that we don't implement or if we raise NotImplementedError.
#   This is sometimes helpful to get increased testing and coverage, but
#   testing takes longer.  Without it, tests will xfail when encountering a
#   function that we don't implement.
#
# Coverage of `nx_cugraph.algorithms` is reported and is a good sanity check
# that algorithms run.

# Warning: cugraph has a .coveragerc file in the <repo root>/python directory,
# so be mindful of its contents and the CWD when running.
# FIXME: should something be added to detect/prevent the above?
set -e
NETWORKX_GRAPH_CONVERT=cugraph \
NETWORKX_TEST_BACKEND=cugraph \
NETWORKX_FALLBACK_TO_NX=True \
    pytest \
    --pyargs networkx \
    --config-file=$(dirname $0)/pyproject.toml \
    --cov-config=$(dirname $0)/pyproject.toml \
    --cov=nx_cugraph \
    --cov-report= \
    "$@"
coverage report \
    --include="*/nx_cugraph/algorithms/*" \
    --omit=__init__.py \
    --show-missing \
    --rcfile=$(dirname $0)/pyproject.toml
