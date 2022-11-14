# Copyright (c) 2022, NVIDIA CORPORATION.
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


def start_server_subprocess(
    host="localhost",
    port=9090,
    graph_creation_extension_dir=None,
    dask_scheduler_file=None,
    env_additions=None,
):
    """
    Start a cugraph_service server as a subprocess. Returns the Popen object
    for the server.
    """
    # Import modules under test here to prevent pytest collection errors if
    # code changes prevent these from being imported.
    # Also check here that cugraph_service_server can be imported
    import cugraph_service_server  # noqa: F401
    from cugraph_service_client import CugraphServiceClient
    from cugraph_service_client.exceptions import CugraphServiceError

    server_process = None
    env_dict = os.environ.copy()
    if env_additions is not None:
        env_dict.update(env_additions)

    # pytest will update sys.path based on the tests it discovers, and for this
    # source tree, an entry for the parent of this "tests" directory will be
    # added. The parent to this "tests" directory also allows imports to find
    # the cugraph_service sources, so in oder to ensure the server that's
    # started is also using the same sources, the PYTHONPATH env should be set
    # to the sys.path being used in this process.
    env_dict["PYTHONPATH"] = ":".join(sys.path)

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

    try:
        server_process = subprocess.Popen(
            args,
            env=env_dict,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        print(
            "\nLaunched cugraph_service server, waiting for it to start...",
            end="",
            flush=True,
        )
        client = CugraphServiceClient(host, port)
        max_retries = 20
        retries = 0
        while retries < max_retries:
            try:
                client.uptime()
                print("started.")
                break
            except CugraphServiceError:
                time.sleep(1)
                retries += 1
        if retries >= max_retries:
            raise RuntimeError("error starting server")
    except Exception:
        if server_process is not None and server_process.poll() is None:
            server_process.terminate()
            server_process.wait(timeout=60)
        raise

    return server_process
