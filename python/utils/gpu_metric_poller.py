# GPUMetricPoller
# Utility class and helpers for retrieving GPU metrics for a specific section of code.
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

import time
import threading
from pynvml import smi

class GPUMetricPoller(threading.Thread):
    """
    Polls smi in a separate thread, saves measurements to instance vars
    """
    def __init__(self, *args, **kwargs):
        self.__stop = False
        super().__init__(*args, **kwargs)
        self.maxGpuUtil = 0
        self.maxGpuMemUsed = 0

    def run(self):
        smi.nvmlInit()
        devObj = smi.nvmlDeviceGetHandleByIndex(0)  # hack - get actual device ID somehow
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

    def stop(self):
        self.__stop = True


def startGpuMetricPolling():
    gpuPollObj = GPUMetricPoller()
    gpuPollObj.start()
    return gpuPollObj


def stopGpuMetricPolling(gpuPollObj):
    gpuPollObj.stop()
    gpuPollObj.join()  # consider using timeout and reporting errors
