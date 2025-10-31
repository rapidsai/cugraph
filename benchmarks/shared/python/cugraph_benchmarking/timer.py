# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import sys
import time


class TimerContext:
    def __init__(self, label="", start_msg="", end_msg="", output_handle=None):
        self.__label = label
        self.__start_msg = start_msg
        self.__end_msg = end_msg
        self.__handle = output_handle or sys.stdout

    def __enter__(self):
        if self.__start_msg:
            output = self.__start_msg
        else:
            output = f"STARTING {self.__label}... "
        self.__handle.write(output)
        self.__handle.flush()
        self.__start_time = time.perf_counter_ns()

    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.perf_counter_ns()
        run_time = (end_time - self.__start_time) / 1e9
        if self.__end_msg:
            output = self.__end_msg + f"{run_time}s\n"
        else:
            output = f"DONE {self.__label}, runtime was: {run_time}s\n"
        self.__handle.write(output)
        self.__handle.flush()
