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

from pathlib import Path

_data_dir = (Path(__file__).parent) / "data"

edgelist_csv_data = {
    "karate": {
        "csv_file_name": (_data_dir / "karate.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "float32"],
        "num_edges": 156,
    },
}

property_csv_data = {
    "merchants": {
        "csv_file_name": (_data_dir / "merchants.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "int32", "float32", "int32", "string"],
        "vert_col_name": "merchant_id",
    },
    "users": {
        "csv_file_name": (_data_dir / "users.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "int32"],
        "vert_col_name": "user_id",
    },
    "transactions": {
        "csv_file_name": (_data_dir / "transactions.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "float32", "float32", "int32", "string"],
        "vert_col_names": ("user_id", "merchant_id"),
    },
    "relationships": {
        "csv_file_name": (_data_dir / "relationships.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "int32"],
        "vert_col_names": ("user_id_1", "user_id_2"),
    },
    "referrals": {
        "csv_file_name": (_data_dir / "referrals.csv").absolute().as_posix(),
        "dtypes": ["int32", "int32", "int32", "int32"],
        "vert_col_names": ("user_id_1", "user_id_2"),
    },
}
