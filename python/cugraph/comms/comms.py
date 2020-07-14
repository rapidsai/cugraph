from cugraph.raft.dask.common.comms import Comms as raftComms
from cugraph.raft.dask.common.comms import worker_state
from cugraph.raft.common.handle import Handle


__instance = None
__default_handle = None


# Intialize Comms. If explicit Comms not provided as arg,
# default Comms are initialized as per client information.
def initialize(arg=None):
    global __instance
    if __instance is None:
        global __default_handle
        __default_handle = None
        if arg is None:
            __instance = raftComms()
            __instance.init()
        else:
            __instance = arg
    else:
        raise Exception("Communicator is already initialized")


# Check is Comms was initialized.
def is_initialized():
    global __instance
    if __instance is not None:
        return True
    else:
        return False


# Get raft Comms
def get_comms():
    global __instance
    return __instance


# Get sessionId for finding sessionstate of workers.
def get_session_id():
    if is_initialized():
        global __instance
        return __instance.sessionId


# Destroy Comms
def destroy():
    global __instance
    if is_initialized():
        __instance.destroy()
        __instance = None

# Default handle in case Comms is not initialized.
# This does not perform nccl initialization.
def get_default_handle():
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
