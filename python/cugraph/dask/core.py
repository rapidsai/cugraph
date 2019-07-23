import logging
import numba.cuda
import random
import time
import os

import cugraph
from dask.distributed import wait, default_client
from threading import Lock, Thread


class IPCThread(Thread):
    """
    This mechanism gets around Numba's restriction of CUDA contexts being
    thread-local by creating a thread that can select its own device.
    This allows the user of IPC handles to open them up directly on the
    same device as the owner (bypassing the need for peer access.)
    """

    def __init__(self, ipcs, device):

        Thread.__init__(self)

        self.lock = Lock()
        self.ipcs = ipcs

        # Use canonical device id
        self.device = get_device_id(device)

        print("Starting new IPC thread on device %i for ipcs %s" %
              (self.device, str(list(ipcs))))
        self.running = False

    def run(self):

        select_device(self.device)

        print("Opening: " + str(self.device) + " "
              + str(numba.cuda.get_current_device()))

        self.lock.acquire()

        try:
            self.arrs = []
            for ipc in self.ipcs:
                self.arrs.append(ipc.open())
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
    return os.environ["CUDA_VISIBLE_DEVICES"].split(",")


def device_of_devicendarray(devicendarray):
    dev = cugraph.device_of_gpu_pointer(devicendarray)
    return get_visible_devices()[dev]


def get_device_id(canonical_name):
    dev_order = get_visible_devices()
    idx = 0
    for dev in dev_order:
        if dev == canonical_name:
            return idx
        idx += 1

    return -1


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port

