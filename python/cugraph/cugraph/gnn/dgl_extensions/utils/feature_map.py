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


def _update_feature_map(
    pg_feature_map, feat_name_obj, contains_vector_features, columns
):
    """
    Update the existing feature map  `pg_feature_map` based on `feat_name_obj`
    """
    if contains_vector_features:
        if feat_name_obj is None:
            raise ValueError(
                "feature name must be provided when wrapping"
                + " multiple columns under a single feature name"
                + " or a feature map"
            )

        if isinstance(feat_name_obj, str):
            pg_feature_map[feat_name_obj] = columns

        elif isinstance(feat_name_obj, dict):
            covered_columns = []
            for col in feat_name_obj.keys():
                current_cols = feat_name_obj[col]
                # Handle strings too
                if isinstance(current_cols, str):
                    current_cols = [current_cols]
                covered_columns = covered_columns + current_cols

            if set(covered_columns) != set(columns):
                raise ValueError(
                    f"All the columns {columns} not covered in {covered_columns} "
                    f"Please check the feature_map {feat_name_obj} provided"
                )

            for key, cols in feat_name_obj.items():
                if isinstance(cols, str):
                    cols = [cols]
                pg_feature_map[key] = cols
        else:
            raise ValueError(f"{feat_name_obj} should be str or dict")
    else:
        if feat_name_obj:
            raise ValueError(
                f"feat_name {feat_name_obj} is only valid when "
                "wrapping multiple columns under feature names"
            )
        for col in columns:
            pg_feature_map[col] = [col]
