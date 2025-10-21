# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Script to be used to simulate a cugraph_service client.
"""
import time
import random

from cugraph_service_client import CugraphServiceClient

client = CugraphServiceClient()

time.sleep(10)
n = int(random.random() * 1000)

# print(f"---> starting {n}", flush=True)

for i in range(1000000):
    extracted_gid = client.extract_subgraph(allow_multi_edges=False)
    # client.delete_graph(extracted_gid)
    # print(f"---> {n}: extracted {extracted_gid}", flush=True)

# print(f"---> done {n}", flush=True)
