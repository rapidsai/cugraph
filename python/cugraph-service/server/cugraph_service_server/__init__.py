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

from cugraph_service_client import defaults
from cugraph_service_client.cugraph_service_thrift import create_server

from cugraph_service_server.cugraph_handler import CugraphHandler


def create_handler(
    graph_creation_extension_dir=None,
    start_local_cuda_cluster=False,
    dask_scheduler_file=None,
):
    """
    Create and return a CugraphHandler instance initialized with
    options. Setting graph_creation_extension_dir to a valid dir results in the
    handler loading graph creation extensions from that dir.
    """
    handler = CugraphHandler()
    if start_local_cuda_cluster and (dask_scheduler_file is not None):
        raise ValueError(
            "dask_scheduler_file cannot be set if start_local_cuda_cluster is True"
        )

    if graph_creation_extension_dir is not None:
        handler.load_graph_creation_extensions(graph_creation_extension_dir)
    if dask_scheduler_file is not None:
        handler.initialize_dask_client(dask_scheduler_file=dask_scheduler_file)
    elif start_local_cuda_cluster:
        handler.initialize_dask_client()

    return handler


def start_server_blocking(
    graph_creation_extension_dir=None,
    start_local_cuda_cluster=False,
    dask_scheduler_file=None,
    host=defaults.host,
    port=defaults.port,
    console_message="",
):
    """
    Start the cugraph_service server on host/port, with graph creation
    extensions in graph_creation_extension_dir preloaded if specified, and a
    dask client initialized based on dask_scheduler_file if specified (if not
    specified, server runs in SG mode). If console_message is specified, the
    string is printed just after the handler is created and before the server
    starts listening for connections. This call blocks indefinitely until
    Ctrl-C.
    """
    handler = create_handler(
        graph_creation_extension_dir, start_local_cuda_cluster, dask_scheduler_file
    )
    if console_message != "":
        print(console_message, flush=True)
    server = create_server(handler, host=host, port=port)
    server.serve()  # blocks until Ctrl-C (kill -2)
