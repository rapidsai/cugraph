# Copyright (c) 2022, NVIDIA CORPORATION.
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

"""
Script to be used to simulate a cugraph_service client.
"""

import random
import time
from pathlib import Path

from cugraph_service_client import CugraphServiceClient


_data_dir = (Path(__file__).parent) / "data"

edgelist_csv_data = {
    "karate": {
        "csv_file_name": (_data_dir / "karate.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "float32"],
        "num_edges": 156,
    },
}

client = CugraphServiceClient()

test_data = edgelist_csv_data["karate"]
client.load_csv_as_edge_data(
    test_data["csv_file_name"],
    dtypes=test_data["dtypes"],
    vertex_col_names=["0", "1"],
    type_name="",
)
time.sleep(10)
n = int(random.random() * 1000)

# print(f"---> starting {n}", flush=True)

for i in range(1000000):
    extracted_gid = client.extract_subgraph(allow_multi_edges=False)
    # client.delete_graph(extracted_gid)
    # print(f"---> {n}: extracted {extracted_gid}", flush=True)

# print(f"---> done {n}", flush=True)
