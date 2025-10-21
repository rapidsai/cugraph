# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest
from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
)

import tempfile

# Avoid timeout during shutdown
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny

import certifi
from ssl import create_default_context
from urllib.request import build_opener, HTTPSHandler, install_opener


# Install SSL certificates
def pytest_sessionstart(session):
    ssl_context = create_default_context(cafile=certifi.where())
    https_handler = HTTPSHandler(context=ssl_context)
    install_opener(build_opener(https_handler))


# module-wide fixtures


@pytest.fixture(scope="module")
def dask_client():
    # start_dask_client will check for the SCHEDULER_FILE and
    # DASK_WORKER_DEVICES env vars and use them when creating a client if
    # set. start_dask_client will also initialize the Comms singleton.
    dask_client, dask_cluster = start_dask_client(
        worker_class=IncreasedCloseTimeoutNanny
    )

    yield dask_client

    stop_dask_client(dask_client, dask_cluster)


# FIXME: Add tests leveraging this fixture
@pytest.fixture(scope="module")
def dask_client_non_p2p():
    # start_dask_client will check for the SCHEDULER_FILE and
    # DASK_WORKER_DEVICES env vars and use them when creating a client if
    # set. start_dask_client will also initialize the Comms singleton.
    dask_client, dask_cluster = start_dask_client(
        worker_class=IncreasedCloseTimeoutNanny, p2p=False
    )

    yield dask_client

    stop_dask_client(dask_client, dask_cluster)


@pytest.fixture(scope="module")
def scratch_dir():
    # This should always be set if doing MG testing, since temporary
    # directories are only accessible from the current process.
    tempdir_object = tempfile.TemporaryDirectory()

    if isinstance(tempdir_object, tempfile.TemporaryDirectory):
        yield tempdir_object.name
    else:
        yield tempdir_object

    del tempdir_object
