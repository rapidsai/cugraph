# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import cupy


from typing import List, Dict


def create_df_from_disjoint_series(series_list: List[cudf.Series]):
    series_list.sort(key=lambda s: len(s), reverse=True)

    df = cudf.DataFrame()
    for s in series_list:
        df[s.name] = s

    return df


def create_df_from_disjoint_arrays(array_dict: Dict[str, cupy.array]):
    series_dict = {}
    for k in list(array_dict.keys()):
        if array_dict[k] is not None:
            series_dict[k] = cudf.Series(array_dict[k], name=k)

    return create_df_from_disjoint_series(list(series_dict.values()))
