
from cugraph.structure.graph_primtypes cimport handle_t
from cugraph.comms.comms cimport init_subcomms as c_init_subcomms


def init_subcomms(handle, row_comm_size):
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t
    c_init_subcomms(handle_[0], row_comm_size)
