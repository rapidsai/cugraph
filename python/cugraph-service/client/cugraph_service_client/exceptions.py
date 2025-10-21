# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph_service_client.cugraph_service_thrift import spec

# FIXME: add more fine-grained exceptions!
CugraphServiceError = spec.CugraphServiceError
