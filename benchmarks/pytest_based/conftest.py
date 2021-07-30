# pytest customizations specific to these benchmarks

def pytest_addoption(parser):
    parser.addoption("--no-rmm-reinit", action="store_true", default=False,
                     help="Do not reinit RMM to run benchmarks with different"
                          " managed memory and pool allocator options.")


def pytest_sessionstart(session):
    # if the --no-rmm-reinit option is given, set (or add to) the CLI "mark
    # expression" (-m) the markers for no managedmem and no poolallocator. This
    # will cause the RMM reinit() function to not be called.
    if session.config.getoption("no_rmm_reinit"):
        newMarkexpr = "managedmem_off and poolallocator_off"
        currentMarkexpr = session.config.getoption("markexpr")

        if ("managedmem" in currentMarkexpr) or \
           ("poolallocator" in currentMarkexpr):
            raise RuntimeError("managedmem and poolallocator markers cannot "
                               "be used with --no-rmm-reinit")

        if currentMarkexpr:
            newMarkexpr = f"({currentMarkexpr}) and ({newMarkexpr})"

        session.config.option.markexpr = newMarkexpr
