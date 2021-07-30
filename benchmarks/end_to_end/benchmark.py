# Copyright (c) 2021, NVIDIA CORPORATION.
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

import time
from functools import wraps


class BenchmarkedResult:
    """
    Class to hold results (the return value of the callable being benchmarked
    and meta-data about the benchmarked function) of a benchmarked function run.
    """
    def __init__(self, name, retval, runtime, params=None):
        self.name = name
        self.retval = retval
        self.runtime = runtime
        self.params = params or {}
        self.validator_result = True


def benchmark(func):
    """
    Returns a callable/closure that wraps func with code to time the func call
    and return a BenchmarkedResult. The resulting callable takes the same
    args/kwargs as func.

    The BenchmarkedResult will have its params value assigned from the kwargs
    dictionary, but the func positional args are not captured. If a user needs
    the params captured for reporting purposes, they must use kwargs.  This is
    useful since positional args can be used for args that would not be
    meaningful in a benchmark result as a param to the benchmark.

    This can be used as a function decorator or a standalone function to wrap
    functions to benchmark.
    """
    benchmark_name = getattr(func, "benchmark_name", func.__name__)
    @wraps(func)
    def benchmark_wrapper(*func_args, **func_kwargs):
        t1 = time.perf_counter()
        retval = func(*func_args, **func_kwargs)
        t2 = time.perf_counter()
        return BenchmarkedResult(name=benchmark_name,
                                 retval=retval,
                                 runtime=(t2-t1),
                                 params=func_kwargs,
                                )

    # Assign the name to the returned callable as well for use in debug prints,
    # etc.
    benchmark_wrapper.name = benchmark_name
    return benchmark_wrapper
