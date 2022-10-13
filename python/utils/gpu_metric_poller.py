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

# GPUMetricPoller
# Utility class and helpers for retrieving GPU metrics for a specific section
# of code.
#
"""
# Example:

# Create/start a GPUMetricPoller object, run code to measure, stop poller:
gpuPollObj = startGpuMetricPolling()
run_cuml_algo(data, **{**param_overrides, **cuml_param_overrides})
stopGpuMetricPolling(gpuPollObj)

# Retrieve measurements from the object:
print("Max GPU memory used: %s" % gpuPollObj.maxGpuMemUsed)
print("Max GPU utilization: %s" % gpuPollObj.maxGpuUtil)
"""

import os
import sys
import threading
from pynvml import smi


class GPUMetricPoller(threading.Thread):
    """
    Polls smi in a forked child process, saves measurements to instance vars
    """

    def __init__(self, *args, **kwargs):
        self.__stop = False
        super().__init__(*args, **kwargs)
        self.maxGpuUtil = 0
        self.maxGpuMemUsed = 0

    @staticmethod
    def __waitForInput(fd):
        # assume non-blocking fd
        while True:
            if not fd.closed:
                line = fd.readline()
                if line:
                    return line
            else:
                break
        return None

    @staticmethod
    def __writeToPipe(fd, strToWrite):
        fd.write(strToWrite)
        fd.flush()

    def __runParentLoop(self, readFileNo, writeFileNo):
        parentReadPipe = os.fdopen(readFileNo)
        parentWritePipe = os.fdopen(writeFileNo, "w")

        self.__writeToPipe(parentWritePipe, "1")
        gpuMetricsStr = self.__waitForInput(parentReadPipe)
        while True:
            # FIXME: this assumes the input received is perfect!
            (memUsed, gpuUtil) = [int(x) for x in gpuMetricsStr.strip().split()]

            if memUsed > self.maxGpuMemUsed:
                self.maxGpuMemUsed = memUsed
            if gpuUtil > self.maxGpuUtil:
                self.maxGpuUtil = gpuUtil

            if not self.__stop:
                self.__writeToPipe(parentWritePipe, "1")
            else:
                self.__writeToPipe(parentWritePipe, "0")
                break
            gpuMetricsStr = self.__waitForInput(parentReadPipe)

        parentReadPipe.close()
        parentWritePipe.close()

    def __runChildLoop(self, readFileNo, writeFileNo):
        childReadPipe = os.fdopen(readFileNo)
        childWritePipe = os.fdopen(writeFileNo, "w")

        smi.nvmlInit()
        # hack - get actual device ID somehow
        devObj = smi.nvmlDeviceGetHandleByIndex(0)
        memObj = smi.nvmlDeviceGetMemoryInfo(devObj)
        utilObj = smi.nvmlDeviceGetUtilizationRates(devObj)
        initialMemUsed = memObj.used
        initialGpuUtil = utilObj.gpu

        controlStr = self.__waitForInput(childReadPipe)
        while True:
            memObj = smi.nvmlDeviceGetMemoryInfo(devObj)
            utilObj = smi.nvmlDeviceGetUtilizationRates(devObj)

            memUsed = memObj.used - initialMemUsed
            gpuUtil = utilObj.gpu - initialGpuUtil

            if controlStr.strip() == "1":
                self.__writeToPipe(childWritePipe, "%s %s\n" % (memUsed, gpuUtil))
            elif controlStr.strip() == "0":
                break
            controlStr = self.__waitForInput(childReadPipe)

        smi.nvmlShutdown()
        childReadPipe.close()
        childWritePipe.close()

    def run(self):
        (parentReadPipeFileNo, childWritePipeFileNo) = os.pipe2(os.O_NONBLOCK)
        (childReadPipeFileNo, parentWritePipeFileNo) = os.pipe2(os.O_NONBLOCK)
        pid = os.fork()
        # parent
        if pid:
            os.close(childReadPipeFileNo)
            os.close(childWritePipeFileNo)
            self.__runParentLoop(parentReadPipeFileNo, parentWritePipeFileNo)

        # child
        else:
            os.close(parentReadPipeFileNo)
            os.close(parentWritePipeFileNo)
            self.__runChildLoop(childReadPipeFileNo, childWritePipeFileNo)
            sys.exit(0)

    def stop(self):
        self.__stop = True


def startGpuMetricPolling():
    gpuPollObj = GPUMetricPoller()
    gpuPollObj.start()
    return gpuPollObj


def stopGpuMetricPolling(gpuPollObj):
    gpuPollObj.stop()
    gpuPollObj.join()  # consider using timeout and reporting errors


"""
smi.nvmlInit()
# hack - get actual device ID somehow
devObj = smi.nvmlDeviceGetHandleByIndex(0)
memObj = smi.nvmlDeviceGetMemoryInfo(devObj)
utilObj = smi.nvmlDeviceGetUtilizationRates(devObj)
initialMemUsed = memObj.used
initialGpuUtil = utilObj.gpu

while not self.__stop:
    time.sleep(0.01)

    memObj = smi.nvmlDeviceGetMemoryInfo(devObj)
    utilObj = smi.nvmlDeviceGetUtilizationRates(devObj)

    memUsed = memObj.used - initialMemUsed
    gpuUtil = utilObj.gpu - initialGpuUtil
    if memUsed > self.maxGpuMemUsed:
        self.maxGpuMemUsed = memUsed
    if gpuUtil > self.maxGpuUtil:
        self.maxGpuUtil = gpuUtil

    smi.nvmlShutdown()
"""


# if __name__ == "__main__":
#     sto=stopGpuMetricPolling
#     po = startGpuMetricPolling()
