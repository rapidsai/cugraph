# Copyright (c) 2024, NVIDIA CORPORATION.
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
Checks if a particular dataset has been downloaded inside the datasets dir
(RAPIDS_DATAEST_ROOT_DIR). If not, the file will be downloaded using the
datasets API.

Positional Arguments:
    1) dataset name (e.g. 'email_Eu_core', 'cit-patents')
       available datasets can be found here: `python/cugraph/cugraph/datasets/__init__.py`
"""

import sys

import cugraph.datasets as cgds


if __name__ == "__main__":
    # download and store dataset (csv) by using the Datasets API
    dataset = sys.argv[1].replace("-", "_")
    dataset_obj = getattr(cgds, dataset)

    if not dataset_obj.get_path().exists():
        dataset_obj.get_edgelist(download=True)
