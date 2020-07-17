from cugraph.raft.dask.common.utils import default_client


def is_worker_organizer(worker_idx):
    return worker_idx == 0


# FIXME: We currently look for the default client from dask, as such is the
# if there is a dask client running without any GPU we will still try
# to run OPG using this client, it also implies that more  work will be
# required  in order to run an OPG Batch in Combination with mutli-GPU Graph
def mg_get_client():
    try:
        client = default_client()
    except ValueError:
        client = None
    return client
