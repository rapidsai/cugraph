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

import argparse
from pathlib import Path

from gaas_client import defaults
from gaas_client.gaas_thrift import create_server
from gaas_server.gaas_handler import GaasHandler


def create_handler(graph_creation_extension_dir=None):
    """
    Create and return a GaasHandler instance initialized with options. Setting
    graph_creation_extension_dir to a valid dir results in the handler loading
    graph creation extensions from that dir.
    """
    handler = GaasHandler()
    if graph_creation_extension_dir:
        handler.load_graph_creation_extensions(graph_creation_extension_dir)
    return handler


def start_server_blocking(handler, host, port):
    """
    Start the GaaS server on host/port, using handler as the request handler
    instance. This call blocks indefinitely until Ctrl-C.
    """
    server = create_server(handler, host=host, port=port)
    server.serve()  # blocks until Ctrl-C (kill -2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="GaaS - (cu)Graph as a Service")
    arg_parser.add_argument("--host",
                            type=str,
                            default=defaults.host,
                            help="hostname the server should use, default " \
                            f"is {defaults.host}")
    arg_parser.add_argument("--port",
                            type=int,
                            default=defaults.port,
                            help="port the server should listen on, default " \
                            f"is {defaults.port}")
    arg_parser.add_argument("--graph-creation-extension-dir",
                            type=Path,
                            help="dir to load graph creation extension " \
                            "functions from")
    args = arg_parser.parse_args()
    handler = create_handler(args.graph_creation_extension_dir)

    print('Starting GaaS...', flush=True)
    start_server_blocking(handler, args.host, args.port)
    print('done.')
