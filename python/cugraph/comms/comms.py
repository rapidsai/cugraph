# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

from cugraph.raft.dask.common.comms import Comms as raftComms
from cugraph.raft.dask.common.comms import worker_state
from cugraph.raft.common.handle import Handle
from cugraph.comms.comms_wrapper import init_subcomms as c_init_subcomms
from dask.distributed import default_client
from cugraph.dask.common import read_utils
import math


__instance = None
__default_handle = None
__subcomm = None


def __get_2D_div(ngpus):
    pcols = int(math.sqrt(ngpus))
    while ngpus % pcols != 0:
        pcols = pcols - 1
    return int(ngpus/pcols), pcols


def subcomm_init(prows, pcols, partition_type):
    sID = get_session_id()
    ngpus = get_n_workers()
    if prows is None and pcols is None:
        if partition_type == 1:
            pcols, prows = __get_2D_div(ngpus)
        else:
            prows, pcols = __get_2D_div(ngpus)
    else:
        if prows is not None and pcols is not None:
            if ngpus != prows*pcols:
                raise Exception('prows*pcols should be equal to the\
 number of processes')
        elif prows is not None:
            if ngpus % prows != 0:
                raise Exception('prows must be a factor of the number\
 of processes')
            pcols = int(ngpus/prows)
        elif pcols is not None:
            if ngpus % pcols != 0:
                raise Exception('pcols must be a factor of the number\
 of processes')
            prows = int(ngpus/pcols)

    client = default_client()
    client.run(_subcomm_init, sID, pcols)
    global __subcomm
    __subcomm = (prows, pcols, partition_type)


def _subcomm_init(sID, partition_row_size):
    handle = get_handle(sID)
    c_init_subcomms(handle, partition_row_size)


def initialize(comms=None,
               p2p=False,
               prows=None,
               pcols=None,
               partition_type=1):
    """
    Initialize a communicator for multi-node/multi-gpu communications.  It is
    expected to be called right after client initialization for running
    multi-GPU algorithms (this wraps raft comms that manages underlying NCCL
    and UCX comms handles across the workers of a Dask cluster).

    It is recommended to also call `destroy()` when the comms are no longer
    needed so the underlying resources can be cleaned up.

    Parameters
    ----------
    comms : raft Comms
        A pre-initialized raft communicator. If provided, this is used for mnmg
        communications. If not provided, default comms are initialized as per
        client information.
    p2p : bool
        Initialize UCX endpoints if True. Default is False.
    prows : int
        Specifies the number of rows when performing a 2D partitioning of the
        input graph. If specified, this must be a factor of the total number of
        parallel processes. When specified with pcols, prows*pcols should be
        equal to the total number of parallel processes.
    pcols : int
        Specifies the number of columns when performing a 2D partitioning of
        the input graph. If specified, this must be a factor of the total
        number of parallel processes. When specified with prows, prows*pcols
        should be equal to the total number of parallel processes.
    partition_type : int
        Valid values are currently 1 or any int other than 1. A value of 1 (the
        default) represents a partitioning resulting in prows*pcols
        partitions. A non-1 value currently results in a partitioning of
        p*pcols partitions, where p is the number of GPUs.
    """

    global __instance
    if __instance is None:
        global __default_handle
        __default_handle = None
        if comms is None:
            # Initialize communicator
            if not p2p:
                raise Exception("Set p2p to True for running mnmg algorithms")
            __instance = raftComms(comms_p2p=p2p)
            __instance.init()
            # Initialize subcommunicator
            subcomm_init(prows, pcols, partition_type)
        else:
            __instance = comms
    else:
        raise Exception("Communicator is already initialized")


def is_initialized():
    """
    Returns True if comms was initialized, False otherwise.
    """
    global __instance
    if __instance is not None:
        return True
    else:
        return False


def get_comms():
    """
    Returns raft Comms instance
    """
    global __instance
    return __instance


def get_workers():
    """
    Returns the workers in the Comms instance, or None if Comms is not
    initialized.
    """
    if is_initialized():
        global __instance
        return __instance.worker_addresses


def get_session_id():
    """
    Returns the sessionId for finding sessionstate of workers, or None if Comms
    is not initialized.
    """
    if is_initialized():
        global __instance
        return __instance.sessionId


def get_2D_partition():
    """
    Returns a tuple representing the 2D partition information: (prows, pcols,
    partition_type)
    """
    global __subcomm
    if __subcomm is not None:
        return __subcomm


def destroy():
    """
    Shuts down initialized comms and cleans up resources.
    """
    global __instance
    if is_initialized():
        __instance.destroy()
        __instance = None


def get_default_handle():
    """
    Returns the default handle. This does not perform nccl initialization.
    """
    global __default_handle
    if __default_handle is None:
        __default_handle = Handle()
    return __default_handle


# Functions to be called from within workers

def get_handle(sID):
    sessionstate = worker_state(sID)
    return sessionstate['handle']


def get_worker_id(sID):
    sessionstate = worker_state(sID)
    return sessionstate['wid']


# FIXME: There are several similar instances of utility functions for getting
# the number of workers, including:
#   * get_n_workers() (from cugraph.dask.common.read_utils)
#   * len(get_visible_devices())
#   * len(numba.cuda.gpus)
# Consider consolidating these or emphasizing why different
# functions/techniques are needed.
def get_n_workers(sID=None):
    if sID is None:
        return read_utils.get_n_workers()
    else:
        sessionstate = worker_state(sID)
        return sessionstate['nworkers']
