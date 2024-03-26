# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

import cupy
import cudf

import warnings


def sampling_results_from_cupy_array_dict(
    cupy_array_dict,
    weight_t,
    num_hops,
    return_offsets=False,
    renumber=False,
):
    """
    Creates a cudf DataFrame from cupy arrays from pylibcugraph wrapper
    """
    results_df = cudf.DataFrame()

    majors = cupy_array_dict["majors"]
    if majors is not None:
        results_df["majors"] = majors

    results_df_cols = [
        "minors",
        "weight",
        "edge_id",
        "edge_type",
    ]

    for col in results_df_cols:
        array = cupy_array_dict[col]
        # The length of each of these arrays should be the same
        results_df[col] = array

    label_hop_offsets = cupy_array_dict["label_hop_offsets"]
    batch_ids = cupy_array_dict["batch_id"]

    if renumber:
        renumber_df = cudf.DataFrame(
            {
                "renumber_map": cupy_array_dict["renumber_map"],
            }
        )

        if not return_offsets:
            if len(batch_ids) > 0:
                batch_ids_r = cudf.Series(batch_ids).repeat(
                    cupy.diff(cupy_array_dict["renumber_map_offsets"])
                )
                batch_ids_r.reset_index(drop=True, inplace=True)
                renumber_df["batch_id"] = batch_ids_r
            else:
                renumber_df["batch_id"] = None

    if return_offsets:
        batches_series = cudf.Series(
            batch_ids,
            name="batch_id",
        )

        offsets_df = cudf.Series(
            label_hop_offsets,
            name="offsets",
        ).to_frame()

        if len(batches_series) > len(offsets_df):
            # this is extremely rare so the inefficiency is ok
            offsets_df = offsets_df.join(batches_series, how="outer").sort_index()
        else:
            offsets_df["batch_id"] = batches_series

        if renumber:
            renumber_offset_series = cudf.Series(
                cupy_array_dict["renumber_map_offsets"], name="renumber_map_offsets"
            )

            if len(renumber_offset_series) > len(offsets_df):
                # this is extremely rare so the inefficiency is ok
                offsets_df = offsets_df.join(
                    renumber_offset_series, how="outer"
                ).sort_index()
            else:
                offsets_df["renumber_map_offsets"] = renumber_offset_series

    else:
        if len(batch_ids) > 0:
            batch_ids_r = cudf.Series(cupy.repeat(batch_ids, num_hops))
            batch_ids_r = cudf.Series(batch_ids_r).repeat(
                cupy.diff(label_hop_offsets)
            )
            batch_ids_r.reset_index(drop=True, inplace=True)

            results_df["batch_id"] = batch_ids_r
        else:
            results_df["batch_id"] = None

    if "majors" not in results_df:
        major_offsets_series = cudf.Series(
            cupy_array_dict["major_offsets"], name="major_offsets"
        )
        if len(major_offsets_series) > len(results_df):
            # this is extremely rare so the inefficiency is ok
            results_df = results_df.join(
                major_offsets_series, how="outer"
            ).sort_index()
        else:
            results_df["major_offsets"] = major_offsets_series

    if return_offsets:
        if renumber:
            return results_df, offsets_df, renumber_df
        else:
            return results_df, offsets_df

    if renumber:
        return results_df, renumber_df

    return (results_df,)
