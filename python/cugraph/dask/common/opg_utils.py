from cugraph.raft.dask.common.comms import Comms
from cugraph.raft.dask.common.utils import default_client


class CommsInitAndDestroyContext:
    def __init__(self, comms):
        self._comms = comms

    def __enter__(self):
        self._comms.init()
        return self

    def __exit__(self, type, value, traceback):
        self._comms.destroy()


def is_worker_organizer(worker_idx):
    return worker_idx == 0


# FIXME: We currently look for the default client from dask, as such is the
# if there is a dask client running without any GPU we will still try
# to run OPG using this client, it also implies that more  work will be
# required  in order to run an OPG Batch in Combination with mutli-GPU Graph
def opg_get_client():
    try:
        client = default_client()
    except ValueError:
        client = None

    return client


def opg_get_comms_using_client(client):
    comms = None

    if client is not None:
        comms = Comms(client=client)

    return comms

