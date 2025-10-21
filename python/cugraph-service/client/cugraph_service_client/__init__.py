# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from cugraph_service_client._version import __git_commit__, __version__
