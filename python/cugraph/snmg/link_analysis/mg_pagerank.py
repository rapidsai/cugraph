# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cugraph.snmg.link_analysis import mg_pagerank_wrapper


def mg_pagerank(src_ptrs_info,
                dest_ptrs_info,
                alpha=0.85,
                max_iter=30):
    df = mg_pagerank_wrapper.mg_pagerank(src_ptrs_info,
                                         dest_ptrs_info,
                                         alpha,
                                         max_iter)

    return df
