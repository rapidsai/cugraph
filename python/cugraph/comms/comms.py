from cugraph.raft.dask.common.comms import Comms as raftComms
from cugraph.raft.dask.common.comms import worker_state

__instance = None

def initialize(arg=None):
    global __instance
    if __instance is None:
       if arg is None:
            __instance = raftComms()
            __instance.init()
       else:
            __instance = arg
    else:
        raise Exception("Communicator is already initialized")

def is_initialized():
    global __instance
    if __instance is not None:
        return True
    else:
        return False

def get_comms():
    global __instance
    return __instance

def destroy():
    if is_initialized():
        __instance.destroy()

def get_session(sessionID):
    return worker_state(sessionID)

