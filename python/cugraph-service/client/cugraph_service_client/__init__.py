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

# constants used by both client and server
# (the server package depends on the client so server code can share client
# code/utilities/defaults/etc.)
supported_extension_return_dtypes = [
    "NoneType",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
]
# make a bi-directional mapping between type strings and ints. This is used for
# sending dtype meta-data between client and server.
extension_return_dtype_map = dict(enumerate(supported_extension_return_dtypes))
extension_return_dtype_map.update(
    dict(map(reversed, extension_return_dtype_map.items()))
)

from cugraph_service_client.client import CugraphServiceClient
from cugraph_service_client.remote_graph import RemoteGraph
