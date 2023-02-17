# Copyright (c) 2023, NVIDIA CORPORATION.
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
# Import FeatureStore class
from cugraph.gnn import FeatureStore
import numpy as np
import cudf
import pytest


def test_feature_storage_from_numpy():
    ar1 = np.random.randint(low=0, high=100, size=100_000)
    ar2 = np.random.randint(low=0, high=100, size=100_000)
    ar3 = np.random.randint(low=0, high=100, size=100_000).reshape(10_000, -1)
    fs = FeatureStore(backend="numpy")
    fs.add_data(ar1, "type1", "feat1")
    fs.add_data(ar2, "type1", "feat2")
    fs.add_data(ar3, "type2", "feat1")

    indices_to_fetch = np.random.randint(low=0, high=len(ar1), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type1", feat_name="feat1")
    expected = ar1[indices_to_fetch]
    np.testing.assert_array_equal(output_fs, expected)

    indices_to_fetch = np.random.randint(low=0, high=len(ar2), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type1", feat_name="feat2")
    expected = ar2[indices_to_fetch]
    np.testing.assert_array_equal(output_fs, expected)

    indices_to_fetch = np.random.randint(low=0, high=len(ar3), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type2", feat_name="feat1")
    expected = ar3[indices_to_fetch]
    np.testing.assert_array_equal(output_fs, expected)


def test_feature_storage_from_cudf():
    ar1 = np.random.randint(low=0, high=100, size=100_000).reshape(10_000, -1)
    df1 = cudf.DataFrame(ar1)
    ar2 = np.random.randint(low=0, high=100, size=100_000).reshape(10_000, -1)
    df2 = cudf.DataFrame(ar2)
    ar3 = np.random.randint(low=0, high=100, size=100_000).reshape(10_000, -1)
    df3 = cudf.DataFrame(ar3)

    fs = FeatureStore(backend="numpy")
    fs.add_data(df1, "type1", "feat1")
    fs.add_data(df2, "type1", "feat2")
    fs.add_data(df3, "type2", "feat1")

    indices_to_fetch = np.random.randint(low=0, high=len(ar1), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type1", feat_name="feat1")
    expected = ar1[indices_to_fetch]
    np.testing.assert_array_equal(output_fs, expected)

    indices_to_fetch = np.random.randint(low=0, high=len(ar2), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type1", feat_name="feat2")
    expected = ar2[indices_to_fetch]
    np.testing.assert_array_equal(output_fs, expected)

    indices_to_fetch = np.random.randint(low=0, high=len(ar3), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type2", feat_name="feat1")
    expected = ar3[indices_to_fetch]
    np.testing.assert_array_equal(output_fs, expected)


def test_feature_storage_pytorch_backend():
    try:
        import torch
    except ModuleNotFoundError:
        pytest.skip("pytorch not available")

    ar1 = np.random.randint(low=0, high=100, size=100_000)
    ar2 = np.random.randint(low=0, high=100, size=100_000)
    ar3 = np.random.randint(low=0, high=100, size=100_000).reshape(-1, 10)
    fs = FeatureStore(backend="torch")
    fs.add_data(ar1, "type1", "feat1")
    fs.add_data(ar2, "type1", "feat2")
    fs.add_data(ar3, "type2", "feat1")

    indices_to_fetch = np.random.randint(low=0, high=len(ar1), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type1", feat_name="feat1")
    expected = ar1[indices_to_fetch]
    assert isinstance(output_fs, torch.Tensor)
    np.testing.assert_array_equal(output_fs.numpy(), expected)

    indices_to_fetch = np.random.randint(low=0, high=len(ar2), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type1", feat_name="feat2")
    expected = ar2[indices_to_fetch]
    assert isinstance(output_fs, torch.Tensor)
    np.testing.assert_array_equal(output_fs.numpy(), expected)

    indices_to_fetch = np.random.randint(low=0, high=len(ar3), size=1024)
    output_fs = fs.get_data(indices_to_fetch, type_name="type2", feat_name="feat1")
    expected = ar3[indices_to_fetch]
    assert isinstance(output_fs, torch.Tensor)
    np.testing.assert_array_equal(output_fs.numpy(), expected)
