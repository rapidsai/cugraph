from itertools import product

import pytest


def genFixtureParamsProduct(*args):
    """
    Returns the cartesian product of the param lists passed in. The lists must
    be flat lists of pytest.param objects, and the result will be a flat list
    of pytest.param objects with values and meta-data combined accordingly. A
    flat list of pytest.param objects is required for pytest fixtures to
    properly recognize the params. The combinations also include ids generated
    from the param values and id names associated with each list. For example:

    genFixtureParamsProduct( ([pytest.param(True, marks=[pytest.mark.A_good]),
                               pytest.param(False, marks=[pytest.mark.A_bad])],
                              "A"),
                             ([pytest.param(True, marks=[pytest.mark.B_good]),
                               pytest.param(False, marks=[pytest.mark.B_bad])],
                              "B") )

    results in fixture param combinations:

    True, True   - marks=[A_good, B_good] - id="A=True,B=True"
    True, False  - marks=[A_good, B_bad]  - id="A=True,B=False"
    False, True  - marks=[A_bad, B_good]  - id="A=False,B=True"
    False, False - marks=[A_bad, B_bad]   - id="A=False,B=False"

    Simply using itertools.product on the lists would result in a list of
    sublists of individual param objects (ie. not "merged"), which would not be
    recognized properly as params for a fixture by pytest.

    NOTE: This function is only needed for parameterized fixtures.
    Tests/benchmarks will automatically get this behavior when specifying
    multiple @pytest.mark.parameterize(param_name, param_value_list)
    decorators.
    """
    # Enforce that each arg is a list of pytest.param objs and separate params
    # and IDs.
    paramLists = []
    ids = []
    paramType = pytest.param().__class__
    for (paramList, id) in args:
        for param in paramList:
            assert isinstance(param, paramType)
        paramLists.append(paramList)
        ids.append(id)

    retList = []
    for paramCombo in product(*paramLists):
        values = [p.values[0] for p in paramCombo]
        marks = [m for p in paramCombo for m in p.marks]
        comboid = ",".join(["%s=%s" % (id, p.values[0])
                            for (p, id) in zip(paramCombo, ids)])
        retList.append(pytest.param(values, marks=marks, id=comboid))
    return retList


# FIXME: write and use mechanism described here for specifying datasets:
#        https://docs.rapids.ai/maintainers/datasets
# FIXME: rlr: soc-twitter-2010.csv crashes with OOM error on my RTX-8000
UNDIRECTED_DATASETS = [
    pytest.param("../datasets/csv/undirected/hollywood.csv",
                 marks=[pytest.mark.small, pytest.mark.undirected]),
    pytest.param("../datasets/csv/undirected/europe_osm.csv",
                 marks=[pytest.mark.undirected]),
    # pytest.param("../datasets/csv/undirected/soc-twitter-2010.csv",
    #              marks=[pytest.mark.undirected]),
]
DIRECTED_DATASETS = [
    pytest.param("../datasets/csv/directed/cit-Patents.csv",
                 marks=[pytest.mark.small, pytest.mark.directed]),
    pytest.param("../datasets/csv/directed/soc-LiveJournal1.csv",
                 marks=[pytest.mark.directed]),
]

MANAGED_MEMORY = [
    pytest.param(True,
                 marks=[pytest.mark.managedmem_on]),
    pytest.param(False,
                 marks=[pytest.mark.managedmem_off]),
]

POOL_ALLOCATOR = [
    pytest.param(True,
                 marks=[pytest.mark.poolallocator_on]),
    pytest.param(False,
                 marks=[pytest.mark.poolallocator_off]),
]

FIXTURE_PARAMS = genFixtureParamsProduct(
    (DIRECTED_DATASETS + UNDIRECTED_DATASETS, "ds"),
    (MANAGED_MEMORY, "mm"),
    (POOL_ALLOCATOR, "pa"))
