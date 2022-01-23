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


# FIXME: add tests for this
cdef assert_success(cugraph_error_code_t code,
                    cugraph_error_t* err,
                    api_name):
    if code != cugraph_error_code_t.CUGRAPH_SUCCESS:
        # FIXME: extract message using cugraph_error_message()
        raise RuntimeError(f"non-success value returned from {api_name}")


cdef assert_CAI_type(obj, var_name, allow_None=False):
    if allow_None:
        if obj is None:
            return
        msg = f"{var_name} must be None or support __cuda_array_interface__"
    else:
        msg = f"{var_name} does not support __cuda_array_interface__"

    if not(hasattr(obj, "__cuda_array_interface__")):
        raise TypeError(msg)
