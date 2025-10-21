# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
