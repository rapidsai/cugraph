# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

# source this script (ie. do not run it) for easier job control from the shell
# FIXME: change this and/or cugraph_service so PYTHONPATH is not needed
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client1_script.py &
sleep 1
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py
