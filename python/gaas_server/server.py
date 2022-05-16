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

from gaas_client import defaults
from gaas_client.gaas_thrift import create_server
from gaas_server.gaas_handler import GaasHandler


def main():
    # FIXME: add CLI options to set non-default host and port values, and
    # possibly other options.
    server = create_server(GaasHandler(),
                           host=defaults.host,
                           port=defaults.port)
    print('Starting the server...')
    server.serve()
    print('done.')


if __name__ == '__main__':
    import sys
    sys.exit(main())
