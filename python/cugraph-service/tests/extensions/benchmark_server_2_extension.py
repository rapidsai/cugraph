# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

from cugraph.experimental import datasets


# Graph creation extensions (these are assumed to return a Graph object)
def create_graph_from_builtin_dataset(dataset_name, mg=False, server=None):
    print("new calling create_graph_from_built_in_dataset: ", dataset_name)
    dataset_obj = getattr(datasets, dataset_name)
    return None
    # return dataset_obj.get_graph(fetch=True)


### Next debugging step... unroll this.  What's in get_graph?
