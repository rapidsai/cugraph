# Copyright (c) 2019, NVIDIA CORPORATION.
#
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
#

import logging
import numba.cuda
import time
import os
from threading import Lock, Thread
import cugraph


class IPCThread(Thread):
    """
    This mechanism gets around Numba's restriction of CUDA contexts being
    thread-local by creating a thread that can select its own device.
    This allows the user of IPC handles to open them up directly on the
    same device as the owner (bypassing the need for peer access.)
    """

    def __init__(self, ipcs, device):
        """
        Initializes the thread with the given IPC handles for the
        given device
        :param ipcs: list[ipc] list of ipc handles with memory on the
                     given device
        :param device: device id to use.
        """

        Thread.__init__(self)

        self.lock = Lock()
        self.ipcs = ipcs

        # Use canonical device id
        self.device = get_device_id(device)

        print("Starting new IPC thread on device %i for ipcs %s" %
              (self.device, str(list(ipcs))))
        self.running = False

    def run(self):
        """
        Starts the current Thread instance enabling memory from the selected
        device to be used.
        """

        select_device(self.device)

        print("Opening: " + str(self.device) + " "
              + str(numba.cuda.get_current_device()))

        self.lock.acquire()

        try:
            self.arrs = [ipc.open() for ipc in self.ipcs]
            self.ptr_info = [x.__cuda_array_interface__ for x in self.arrs]

            self.running = True
        except Exception as e:
            logging.error("Error opening ipc_handle on device " +
                          str(self.device) + ": " + str(e))

        self.lock.release()

        while (self.running):
            time.sleep(0.0001)

        try:
            logging.warn("Closing: " + str(self.device) +
                         str(numba.cuda.get_current_device()))
            self.lock.acquire()
            [ipc.close() for ipc in self.ipcs]
            self.lock.release()

        except Exception as e:
            logging.error("Error closing ipc_handle on device " +
                          str(self.device) + ": " + str(e))

    def close(self):

        """
        This should be called before calling join(). Otherwise, IPC handles
        may not be properly cleaned up.
        """
        self.lock.acquire()
        self.running = False
        self.lock.release()

    def info(self):
        """
        Warning: this method is invoked from the calling thread. Make
        sure the context in the thread reading the memory is tied to
        self.device, otherwise an expensive peer access might take
        place underneath.
        """
        while (not self.running):
            time.sleep(0.0001)

        return self.ptr_info


def new_ipc_thread(ipcs, dev):
    t = IPCThread(ipcs, dev)
    t.start()
    return t


def select_device(dev, close=True):
    """
    Use numbas numba to select the given device, optionally
    closing and opening up a new cuda context if it fails.
    :param dev: int device to select
    :param close: bool close the cuda context and create new one?
    """
    if numba.cuda.get_current_device().id != dev:
        logging.warn("Selecting device " + str(dev))
        if close:
            numba.cuda.close()
        numba.cuda.select_device(dev)
        if dev != numba.cuda.get_current_device().id:
            logging.warn("Current device " +
                         str(numba.cuda.get_current_device()) +
                         " does not match expected " + str(dev))


def get_visible_devices():
    """
    Return a list of the CUDA_VISIBLE_DEVICES
    :return: list[int] visible devices
    """
    # TODO: Shouldn't have to split on every call
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def device_of_devicendarray(devicendarray):
    """
    Returns the device that backs memory allocated on the given
    deviceNDArray
    :param devicendarray: devicendarray array to check
    :return: int device id
    """
    dev = cugraph.device_of_gpu_pointer(devicendarray)
    return get_visible_devices()[dev]


def get_device_id(canonical_name):
    """
    Given a local device id, find the actual "global" id
    :param canonical_name: the local device name in CUDA_VISIBLE_DEVICES
    :return: the global device id for the system
    """
    dev_order = get_visible_devices()
    idx = 0
    for dev in dev_order:
        if dev == canonical_name:
            return idx
        idx += 1

    return -1


def parse_host_port(address):
    """
    Given a string address with host/port, build a tuple(host, port)
    :param address: string address to parse
    :return: tuple(host, port)
    """
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port
