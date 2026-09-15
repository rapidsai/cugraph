# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# GPUMetricPoller
# Utility class and helpers for retrieving GPU metrics for a specific section
# of code.
#
# Requires:
#  cuda_core >= 1.0.0
#  cuda_bindings >= 12.9.6 or >= 13.2.0

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


from cuda.core import system


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

        # hack - get actual device ID somehow
        devObj = system.Device(index=0)
        memObj = devObj.memory_info
        utilObj = devObj.utilization
        initialMemUsed = memObj.used
        initialGpuUtil = utilObj.gpu

        controlStr = self.__waitForInput(childReadPipe)
        while True:
            memObj = devObj.memory_info
            utilObj = devObj.utilization

            memUsed = memObj.used - initialMemUsed
            gpuUtil = utilObj.gpu - initialGpuUtil

            if controlStr.strip() == "1":
                self.__writeToPipe(childWritePipe, "%s %s\n" % (memUsed, gpuUtil))
            elif controlStr.strip() == "0":
                break
            controlStr = self.__waitForInput(childReadPipe)

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
