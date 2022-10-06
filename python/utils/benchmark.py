# Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

# from time import process_time_ns   # only in 3.7!
from time import clock_gettime, CLOCK_MONOTONIC_RAW

import numpy as np

from gpu_metric_poller import startGpuMetricPolling, stopGpuMetricPolling


class Benchmark:

    resultsDict = {}
    metricNameCellWidth = 20
    valueCellWidth = 40

    def __init__(self, func, name="", args=None):
        """
        func = the callable to wrap
        name = name of callable, needed mostly for bookkeeping
        args = args to pass the callable (default is no args)
        """
        self.func = func
        self.name = name or func.__name__
        self.args = args or ()

    def run(self, n=1):
        """
        Run self.func() n times and compute the average of all runs for all
        metrics after discarding the min and max values for each.
        """
        retVal = None
        # Return or create the results dict unique to the function name
        funcResultsDict = self.resultsDict.setdefault(self.name, {})

        # FIXME: use a proper logger
        print("Running %s" % self.name, end="", flush=True)

        try:
            exeTimes = []
            gpuMems = []
            gpuUtils = []

            if n > 1:
                print("  - iteration ", end="", flush=True)

            for i in range(n):
                if n > 1:
                    print(i + 1, end="...", flush=True)
                gpuPollObj = startGpuMetricPolling()
                # st = process_time_ns()
                st = clock_gettime(CLOCK_MONOTONIC_RAW)
                retVal = self.func(*self.args)
                stopGpuMetricPolling(gpuPollObj)

                # exeTime = (process_time_ns() - st) / 1e9
                exeTime = clock_gettime(CLOCK_MONOTONIC_RAW) - st
                exeTimes.append(exeTime)
                gpuMems.append(gpuPollObj.maxGpuUtil)
                gpuUtils.append(gpuPollObj.maxGpuMemUsed)

            print("  - done running %s." % self.name, flush=True)

        except Exception as e:
            funcResultsDict["ERROR"] = str(e)
            print(
                "   %s | %s"
                % (
                    "ERROR".ljust(self.metricNameCellWidth),
                    str(e).ljust(self.valueCellWidth),
                )
            )
            stopGpuMetricPolling(gpuPollObj)
            return

        funcResultsDict["exeTime"] = self.__computeValue(exeTimes)
        funcResultsDict["maxGpuUtil"] = self.__computeValue(gpuMems)
        funcResultsDict["maxGpuMemUsed"] = self.__computeValue(gpuUtils)

        for metricName in ["exeTime", "maxGpuUtil", "maxGpuMemUsed"]:
            val = funcResultsDict[metricName]
            print(
                "   %s | %s"
                % (
                    metricName.ljust(self.metricNameCellWidth),
                    str(val).ljust(self.valueCellWidth),
                ),
                flush=True,
            )

        return retVal

    def __computeValue(self, vals):
        """
        Return the avergage val from the list of vals filtered to remove 2
        std-deviations from the original average.
        """
        avg = np.mean(vals)
        std = np.std(vals)
        filtered = [x for x in vals if ((avg - (2 * std)) <= x <= (avg + (2 * std)))]
        if len(filtered) != len(vals):
            print("filtered outliers: %s" % (set(vals) - set(filtered)))

        return np.average(filtered)
