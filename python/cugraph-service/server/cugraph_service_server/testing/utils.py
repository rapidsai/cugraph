# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import subprocess
import time
from tempfile import TemporaryDirectory
from pathlib import Path

from cugraph_service_client import (
    CugraphServiceClient,
    defaults,
)
from cugraph_service_client.exceptions import CugraphServiceError


def create_tmp_extension_dir(file_contents, file_name="my_extension.py"):
    """
    Create and return a temporary dir to be used as a dir containing extensions
    to be read by a cugraph_service server. file_contents is a string
    containing the contents of the extension file to write out.
    """
    tmp_extension_dir = TemporaryDirectory()

    graph_creation_extension_file = open(Path(tmp_extension_dir.name) / file_name, "w")
    print(file_contents, file=graph_creation_extension_file, flush=True)

    return tmp_extension_dir


_failed_subproc_err_msg = (
    f"\n{'*' * 21} cugraph-service-server STDOUT/STDERR {'*' * 22}\n"
    "%s"
    f"{'*' * 80}\n"
)


def start_server_subprocess(
    host=defaults.host,
    port=defaults.port,
    graph_creation_extension_dir=None,
    start_local_cuda_cluster=False,
    dask_scheduler_file=None,
    env_additions=None,
):
    """
    Start a cugraph_service server as a subprocess. Returns the Popen object
    for the server.
    """
    server_process = None
    env_dict = os.environ.copy()
    if env_additions is not None:
        env_dict.update(env_additions)

    # pytest will update sys.path based on the tests it discovers and optional
    # settings in pytest.ini. Make sure any path settings are passed on the the
    # server so modules are properly found.
    env_dict["PYTHONPATH"] = ":".join(sys.path)

    # special case: some projects organize their tests/benchmarks by package
    # name, such as "cugraph". Unfortunately, these can collide with installed
    # package names since python will treat them as a namespace package if this
    # is run from a directory with a "cugraph" or similar subdir. Simply change
    # to a temp dir prior to running the server to avoid collisions.
    tempdir_object = TemporaryDirectory()

    args = [
        sys.executable,
        "-m",
        "cugraph_service_server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if graph_creation_extension_dir is not None:
        args += [
            "--graph-creation-extension-dir",
            graph_creation_extension_dir,
        ]
    if dask_scheduler_file is not None:
        args += [
            "--dask-scheduler-file",
            dask_scheduler_file,
        ]
    if start_local_cuda_cluster:
        args += ["--start-local-cuda-cluster"]

    try:
        print(
            f"Starting server subprocess using:\n\"{' '.join(args)}\"\n",
            flush=True,
        )
        server_process = subprocess.Popen(
            args,
            env=env_dict,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=tempdir_object.name,
        )

        # Attach tempdir_object to server_process so it is not deleted and
        # removed when it goes out of scope. Instead, it should get deleted
        # when server_process is GC'd
        server_process.tempdir = tempdir_object

        print(
            "Launched cugraph_service server, waiting for it to start...",
            end="",
            flush=True,
        )
        client = CugraphServiceClient(host, port)
        max_retries = 60
        retries = 0
        while retries < max_retries:
            try:
                client.uptime()
                print("started.")
                break
            except CugraphServiceError:
                time.sleep(1)
                retries += 1

            # poll() returns exit code, or None if still running
            if (server_process is not None) and (server_process.poll() is not None):
                err_output = _failed_subproc_err_msg % server_process.stdout.read()
                server_process = None
                raise RuntimeError(f"error starting server: {err_output}")

            if retries >= max_retries:
                raise RuntimeError("timed out waiting for server to respond")

    except Exception:
        # Stop the server if still running
        if server_process is not None:
            if server_process.poll() is None:
                server_process.terminate()
                server_process.wait(timeout=60)
            print(_failed_subproc_err_msg % server_process.stdout.read())

        raise

    return server_process


def ensure_running_server(
    host=defaults.host,
    port=defaults.port,
    dask_scheduler_file=None,
    start_local_cuda_cluster=False,
):
    """
    Returns a tuple containg a CugraphService instance and a Popen instance for
    the server subprocess. If a server is found running, the Popen instance
    will be None and no attempt will be made to start a subprocess.
    """
    host = "localhost"
    port = 9090
    client = CugraphServiceClient(host, port)
    server_process = None

    try:
        client.uptime()
        print("FOUND RUNNING SERVER, ASSUMING IT SHOULD BE USED FOR TESTING!")

    except CugraphServiceError:
        # A server was not found, so start one for testing then stop it when
        # testing is done.
        server_process = start_server_subprocess(
            host=host,
            port=port,
            start_local_cuda_cluster=start_local_cuda_cluster,
            dask_scheduler_file=dask_scheduler_file,
        )

    return (client, server_process)


def ensure_running_server_for_sampling(
    host=defaults.host,
    port=defaults.port,
    dask_scheduler_file=None,
    start_local_cuda_cluster=False,
):
    """
    Returns a tuple containing a Popen object for the running cugraph-service
    server subprocess, and a client object connected to it.  If a server was
    detected already running, the Popen object will be None.
    """
    (client, server_process) = ensure_running_server(
        host, port, dask_scheduler_file, start_local_cuda_cluster
    )

    # Ensure the extensions needed for these benchmarks are loaded
    required_graph_creation_extension_module = "benchmark_server_extension"
    server_data = client.get_server_info()
    # .stem excludes .py extensions, so it can match a python module name
    loaded_graph_creation_extension_modules = [
        Path(m).stem for m in server_data["graph_creation_extensions"]
    ]
    if (
        required_graph_creation_extension_module
        not in loaded_graph_creation_extension_modules
    ):
        modules_loaded = client.load_graph_creation_extensions(
            "cugraph_service_server.testing.benchmark_server_extension"
        )
        if len(modules_loaded) < 1:
            raise RuntimeError(
                "failed to load graph creation extension "
                f"{required_graph_creation_extension_module}"
            )

    loaded_extension_modules = [Path(m).stem for m in server_data["extensions"]]
    if required_graph_creation_extension_module not in loaded_extension_modules:
        modules_loaded = client.load_extensions(
            "cugraph_service_server.testing.benchmark_server_extension"
        )
        if len(modules_loaded) < 1:
            raise RuntimeError(
                "failed to load extension "
                f"{required_graph_creation_extension_module}"
            )

    return (client, server_process)
