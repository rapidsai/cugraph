import cugraph.comms.comms as Comms

from cugraph.comms.comms cimport key_naming_t, subcomm_factory_t


def init_subcomms(handle, row_comm_size):
    """
    cdef partition_2d::subcomm_factory_t<partition_2d::key_naming_t, int> subcomm_factory(handle,
                                                                                   row_comm_size)
    """
    print("IN SUBCOMMS WRAPPER")
    cdef subcomm_factory_t[key_naming_t, int] subcomm_factory = subcomm_factory_t[key_naming_t, int](handle, row_comm_size)
    print("DONE IN SUBCOMMS WRAPPER")
