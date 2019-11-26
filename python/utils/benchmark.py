import os
import sys
# from time import process_time_ns   # only in 3.7!
from time import clock_gettime, CLOCK_MONOTONIC_RAW

from gpu_metric_poller import startGpuMetricPolling, stopGpuMetricPolling


class Nop:
    def __getattr__(self, attr):
        return Nop()

    def __getitem__(self, key):
        return Nop()

    def __call__(self, *args, **kwargs):
        return Nop()


nop = Nop()


# wrappers
def noStdoutWrapper(func):
    def wrapper(*args):
        prev = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            retVal = func(*args)
            sys.stdout = prev
            return retVal
        except Exception:
            sys.stdout = prev
            raise
    return wrapper


def logExeTime(func, name=""):
    def wrapper(*args):
        retVal = None
        # Return or create the results dict for the function name
        perfData = logExeTime.perfData.setdefault(name or func.__name__, {})
        try:
            # st = process_time_ns()
            st = clock_gettime(CLOCK_MONOTONIC_RAW)
            retVal = func(*args)
        except Exception as e:
            perfData["ERROR"] = str(e)
            return
        # exeTime = (process_time_ns() - st) / 1e9
        exeTime = clock_gettime(CLOCK_MONOTONIC_RAW) - st
        perfData["exeTime"] = exeTime
        return retVal
    return wrapper


logExeTime.perfData = {}


def logGpuMetrics(func, name=""):
    def wrapper(*args):
        retVal = None
        # Return or create the results dict for the function name
        perfData = logGpuMetrics.perfData.setdefault(name or func.__name__, {})

        try:
            gpuPollObj = startGpuMetricPolling()
            retVal = func(*args)
            stopGpuMetricPolling(gpuPollObj)
        except Exception as e:
            perfData["ERROR"] = str(e)
            return
        perfData["maxGpuUtil"] = gpuPollObj.maxGpuUtil
        perfData["maxGpuMemUsed"] = gpuPollObj.maxGpuMemUsed
        return retVal
    return wrapper


logGpuMetrics.perfData = {}


def printLastResult(func, name=""):
    def wrapper(*args):
        retVal = func(*args)
        funcNames = printLastResult.perfData.keys()
        diff = set(funcNames) - printLastResult.funcsPrinted
        if diff:
            metricNameWidth = printLastResult.metricNameCellWidth
            valWidth = printLastResult.valueCellWidth
            funcName = diff.pop()
            valDict = printLastResult.perfData[funcName]
            print(funcName)
            for metricName in sorted(valDict.keys()):
                val = valDict[metricName]
                print("   %s | %s" % (metricName.ljust(metricNameWidth),
                                      str(val).ljust(valWidth)))
            printLastResult.funcsPrinted = set(funcNames)
        return retVal
    return wrapper


printLastResult.perfData = {}
printLastResult.metricNameCellWidth = 20
printLastResult.valueCellWidth = 40
printLastResult.funcsPrinted = set()


class WrappedFunc:
    wrappers = []

    def __init__(self, func, name="", args=None, extraRunWrappers=None):
        """
        func = the callable to wrap
        name = name of callable, needed mostly for bookkeeping
        args = args to pass the callable (default is no args)
        extraRunWrappers = list of functions that return a callable, used for
           wrapping the callable further to modify its environment, add timers,
           log calls, etc.
        """
        self.func = func
        self.name = name or func.__name__
        self.args = args or ()
        runWrappers = (extraRunWrappers or []) + self.wrappers

        # The callable is the callable obj returned after all wrappers applied
        for wrapper in runWrappers:
            self.func = wrapper(self.func, self.name)
            self.func.__name__ = self.name

    def run(self):
        return self.func(*self.args)


class Benchmark(WrappedFunc):
    wrappers = [logExeTime, logGpuMetrics, printLastResult]
