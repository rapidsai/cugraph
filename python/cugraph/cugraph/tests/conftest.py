# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from ssl import create_default_context
from urllib.request import build_opener, HTTPSHandler, install_opener

import packaging.version
import pytest
import networkx as nx
import certifi
from dask_cuda.utils_test import (
    IncreasedCloseTimeoutNanny,
)  # Avoid timeout during shutdown

from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
)


# Hooks
# =============================================================================

# Install SSL certificates
def pytest_sessionstart(session):
    ssl_context = create_default_context(cafile=certifi.where())
    https_handler = HTTPSHandler(context=ssl_context)
    install_opener(build_opener(https_handler))


def pytest_collection_modifyitems(config, items):
    """Modify pytest items after tests have been collected."""
    installed_nx_version = packaging.version.parse(nx.__version__)
    for item in items:
        # Skip tests marked as requiring a specific version of NetworkX if
        # the installed version is too old
        for mark in item.iter_markers(name="requires_nx"):
            ver_str = mark.kwargs.get(
                "version", mark.args[0] if len(mark.args) > 0 else None
            )
            if ver_str is None:
                raise TypeError("requires_nx marker must specify a version")
            min_required_nx_version = packaging.version.parse(ver_str)
            if installed_nx_version < min_required_nx_version:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"Requires networkx >= {min_required_nx_version}, "
                            f"(version installed: {installed_nx_version})"
                        )
                    )
                )


# Fixtures
# =============================================================================


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
