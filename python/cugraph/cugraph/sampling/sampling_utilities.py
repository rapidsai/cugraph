# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy
import cudf


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

    major_col_name = "majors"

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
        if col in cupy_array_dict.keys():
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
                batch_ids_r = cudf.Series(cupy.unique(batch_ids)).repeat(
                    cupy.diff(cupy_array_dict["renumber_map_offsets"])
                )
                batch_ids_r.reset_index(drop=True, inplace=True)
                renumber_df["batch_id"] = batch_ids_r
            else:
                renumber_df["batch_id"] = None

    if return_offsets:
        batches_series = cudf.Series(
            cupy.unique(batch_ids),
            name="batch_id",
        )

        offsets_df = cudf.Series(
            label_hop_offsets[cupy.arange(len(cupy.unique(batch_ids)) + 1) * num_hops],
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
            results_df["batch_id"] = batch_ids
        else:
            results_df["batch_id"] = None

    if len(batch_ids) > 0:
        hop_ids_r = cudf.Series(cupy.arange(num_hops))
        hop_ids_r = cudf.concat(
            [hop_ids_r] * len(cudf.Series(batch_ids).unique()), ignore_index=True
        )

        # generate the hop column
        hop_ids_r = (
            cudf.Series(hop_ids_r, name="hop_id")
            .repeat(cupy.diff(label_hop_offsets))
            .reset_index(drop=True)
        )
    else:
        hop_ids_r = cudf.Series(name="hop_id", dtype="int32")

    results_df = results_df.join(hop_ids_r, how="outer").sort_index()

    if major_col_name not in results_df:
        major_offsets_series = cudf.Series(
            cupy_array_dict["major_offsets"], name="major_offsets"
        )
        if len(major_offsets_series) > len(results_df):
            # this is extremely rare so the inefficiency is ok
            results_df = results_df.join(major_offsets_series, how="outer").sort_index()
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


def legacy_sampling_results_from_cupy_array_dict(
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

    major_col_name = "majors"
    minor_col_name = "minors"

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

    results_df.rename(
        columns={"majors": major_col_name, "minors": minor_col_name}, inplace=True
    )

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
            batch_ids_r = cudf.Series(batch_ids_r).repeat(cupy.diff(label_hop_offsets))
            batch_ids_r.reset_index(drop=True, inplace=True)

            results_df["batch_id"] = batch_ids_r
        else:
            results_df["batch_id"] = None

    if len(batch_ids) > 0:
        hop_ids_r = cudf.Series(cupy.arange(num_hops))
        hop_ids_r = cudf.concat([hop_ids_r] * len(batch_ids), ignore_index=True)

        # generate the hop column
        hop_ids_r = (
            cudf.Series(hop_ids_r, name="hop_id")
            .repeat(cupy.diff(label_hop_offsets))
            .reset_index(drop=True)
        )
    else:
        hop_ids_r = cudf.Series(name="hop_id", dtype="int32")

    results_df = results_df.join(hop_ids_r, how="outer").sort_index()

    if major_col_name not in results_df:
        major_offsets_series = cudf.Series(
            cupy_array_dict["major_offsets"], name="major_offsets"
        )
        if len(major_offsets_series) > len(results_df):
            # this is extremely rare so the inefficiency is ok
            results_df = results_df.join(major_offsets_series, how="outer").sort_index()
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
