# Security Policy

cuGraph is a GPU-accelerated graph analytics library — a C++/CUDA core
(`libcugraph`), a stable C ABI (`cpp/src/c_api/`), Cython bindings
(`pylibcugraph`), Python APIs (`cugraph`, `cugraph.gnn`), and multi-node
distributed estimators built on Dask, RAFT, and UCX (`cugraph.dask`,
`raft-dask`, `ucxx`). It is a library, not a service: it runs in-process
inside a Python interpreter, a Dask worker, or a native C/C++/Go application,
and inherits the caller's privilege.

Its security posture is shaped by the inputs it ingests — vertex and edge
arrays, serialized graph topology, and (on the test/benchmark side)
artifacts fetched from the network — and by the C / Cython / Python / Dask
boundaries it crosses.

## Reporting a Vulnerability

Please report security vulnerabilities privately through one of the channels
below. **Do not open a public GitHub issue, PR, or discussion** for a
suspected vulnerability.

1. **NVIDIA Vulnerability Disclosure Program (preferred)**
   <https://www.nvidia.com/en-us/security/>
   Submit through the NVIDIA PSIRT web form. This is the fastest path to
   triage and tracking.

2. **Email NVIDIA PSIRT**
   psirt@nvidia.com — encrypt sensitive reports with the
   [NVIDIA PSIRT PGP key](https://www.nvidia.com/en-us/security/pgp-key).

3. **GitHub Private Vulnerability Reporting**
   Use the **Security** tab on this repository → *Report a vulnerability*.

Please include, where possible:

- Affected component (e.g. a specific algorithm in `cpp/src/`, the C API,
  Cython bindings, `cugraph.dask`, the dataset downloader)
- cuGraph / libcugraph version, CUDA version, GPU model, and OS
- Reproduction steps and a minimal proof-of-concept (PoC) input or graph
- Impact assessment (memory corruption, code execution, DoS, info disclosure)
- Any relevant CWE / CVE identifiers

NVIDIA PSIRT will acknowledge receipt and coordinate triage, fix development,
and coordinated disclosure. More on NVIDIA's response process:
<https://www.nvidia.com/en-us/security/psirt-policies/>.

## Security Architecture & Context

**Classification:** Library (C++/CUDA core with a public C ABI, plus Cython
and Python bindings; distributed estimators via Dask + UCX).

**Primary security responsibility:** Safely ingest graph topology and
attribute arrays, run analytics and GNN-related primitives on the GPU, and
hand results back to the caller without crashing the host process,
corrupting memory, or mis-reinterpreting types across the language
boundary.

**Components and trust boundaries:**

- **libcugraph** (`cpp/src/`) — C++/CUDA algorithm core. Major subsystems:
  `centrality/`, `community/`, `components/`, `cores/`, `layout/`,
  `linear_assignment/` (Hungarian and successors), `link_analysis/`,
  `link_prediction/`, `sampling/`, `structure/`, `traversal/`, `tree/`,
  `generators/`, plus the multi-threaded / multi-GPU subsystem `mtmg/`.
- **C API** (`cpp/src/c_api/`) — stable C ABI used by external bindings
  (e.g. Go, additional language wrappers). Callers across this boundary
  must respect the documented type, ownership, and lifetime contracts.
- **libcugraph_etl** (`cpp/libcugraph_etl/`) — ETL helpers such as graph
  renumbering for ingest pipelines.
- **pylibcugraph** (`python/pylibcugraph/`) — Cython bindings; the primary
  surface that translates Python array shapes and dtypes into the C++/CUDA
  layer.
- **cugraph** (`python/cugraph/cugraph/`) — Python API and analytics
  pipelines, including `cugraph.gnn` (graph neural network primitives,
  used by external GNN frameworks).
- **`cugraph.dask`** — multi-GPU / multi-node distributed analytics built
  on Dask, `raft-dask`, and `ucxx`. Cluster communication uses Dask's
  pickle-based serialization plus UCX transport.
- **`cugraph.datasets`** (`python/cugraph/cugraph/datasets/`) — benchmark
  and example dataset registry. Resolves dataset URLs from
  `datasets_config.yaml` and fetches them via `fsspec[http]`.
- **`cugraph.testing`** — test result loading helpers, which historically
  used `tarfile.extractall` without member filtering when unpacking
  cached result archives.

**Out of scope for this policy:** vulnerabilities in CUDA, the NVIDIA
driver, RAFT/pylibraft, UCX/UCXX, cuDF, RMM, Dask, fsspec, or the JVM.
Report those to their respective projects (NVIDIA driver and CUDA bugs
still go to PSIRT).

## Threat Model

The threats below trace to specific components in this repository. Several
have already been observed and remediated through the
[RAPIDS Security Audit](https://github.com/orgs/rapidsai/projects/207); they
are listed so that callers and integrators understand the classes of bugs
the library defends against.

1. **Type confusion across the C / Cython / C++ boundary.**
   Algorithms receive vertex/edge arrays as untyped pointers with
   separately-supplied dtype metadata. A mismatch — for example, an int32
   buffer reinterpreted as a `double*` — has driven memory corruption in
   the past (the dense Hungarian solver in `cpp/src/linear_assignment/`).
   Any caller crossing the C ABI or the Cython boundary with malformed
   dtype metadata can drive the same class of bug elsewhere.

2. **Insufficient input validation in the Python / Cython bindings.**
   The bindings in `python/pylibcugraph/` and `python/cugraph/` historically
   accept caller-supplied vertex IDs, edge arrays, dimension counts, and
   parameter values without consistently checking ranges, sign,
   monotonicity, or self-consistency before passing them into kernels.
   A pathological combination of values can drive OOB indexing, integer
   overflow, or divide-by-zero inside CUDA kernels downstream.

3. **Hostile graph topology causing memory corruption or DoS.**
   Graph algorithms in `cpp/src/` (sampling, centrality, community,
   traversal) assume edge lists reference in-range vertex IDs and that
   COO/CSR offsets are monotonic and bounded. A topology file with
   out-of-range vertices, non-monotonic offsets, or `n*m`-product
   overflow in shape arithmetic can drive OOB reads or pathological
   allocation patterns.

4. **Pickle deserialization on distributed paths.**
   `cugraph.dask` and `raft-dask` participate in Dask Distributed's
   pickle-based RPC. Loading pickled cuGraph state — or accepting work
   from an unauthenticated Dask scheduler — is equivalent to arbitrary
   code execution by design of the format.

5. **Untrusted dataset downloads (test / benchmark surface).**
   `cugraph.datasets.Dataset.get_path()` fetches files referenced in
   `datasets_config.yaml` via fsspec. The downloader does not enforce
   TLS-only URLs or verify checksums / signatures of downloaded
   artifacts, so a network adversary or compromised mirror can substitute
   graph data ingested by tests and benchmarks.

6. **Archive extraction without member filtering.**
   Test-result helpers under `python/cugraph/cugraph/testing/` historically
   call `tarfile.extractall` without a member filter, which allows path
   traversal (absolute paths, `..` components, or symlinks) on a hostile
   archive — leading to file overwrite outside the extraction directory.

7. **Shell injection in CI and development scripts.**
   Several `subprocess.run(..., shell=True)` call sites in CI helpers,
   build scripts, and development utilities interpolate variables into a
   shell command string. If any of those variables come from an
   attacker-influenced environment (a PR title, a branch name, a forked
   workflow input), they yield arbitrary command execution in the CI
   runner or developer shell. This is not part of the runtime attack
   surface, but it is part of the repository's supply-chain surface.

8. **UCX / ucxx cluster transport.**
   `cugraph.dask` uses UCX (via ucxx) for high-performance inter-worker
   communication. UCX is not authenticated and assumes a trusted
   network fabric; placing UCX traffic on an untrusted network exposes
   both data and the pickle-based control channel.

## Critical Security Assumptions

cuGraph is a library and inherits the caller's privilege; the following are
assumed of the caller / deployer.

- **Graph inputs are well-formed.**
  cuGraph assumes caller-supplied vertex IDs, edge arrays, COO/CSR
  offsets, and weight arrays have valid dtypes, in-range values, and
  internally consistent sizes. Callers ingesting graphs from external
  sources should validate `dtype`, vertex range, offset monotonicity,
  and overall topology before passing it to an algorithm.

- **Dtype metadata accompanying untyped buffers is honest.**
  The C ABI and Cython bindings trust the dtype tag the caller supplies
  alongside a vertex or edge buffer. A caller passing a buffer of one
  dtype with a tag for another defeats cuGraph's internal type
  dispatch.

- **Resource limits are imposed externally.**
  cuGraph does not cap memory or time per call. Many algorithms have
  super-linear cost in `|V|`, `|E|`, or `|V|·|E|`. Callers operating on
  untrusted graph inputs should run cuGraph in a process with
  cgroup / ulimit / container memory and CPU limits, and should bound
  the topology size they accept.

- **Distributed cluster peers are mutually trusted.**
  `cugraph.dask`, `raft-dask`, and UCX-based transport assume mutually
  authenticated peers on a private network. Pickle-based serialization
  used over Dask's control channel is unsafe across trust boundaries.

- **Transport security is provided externally.**
  cuGraph does not implement TLS, authentication, or authorization for
  any network use (Dask cluster traffic, UCX, fsspec dataset downloads).
  Confidentiality and integrity depend on the surrounding stack — TLS in
  fsspec's HTTP backend, UCX's deployment topology, Dask's TLS
  configuration, or the caller's networking.

- **Benchmark and example datasets are not security-critical inputs.**
  The `cugraph.datasets` registry is intended for tests, examples, and
  benchmarks. Production users should not rely on it as a trusted data
  source, and the registry's lack of integrity verification should not be
  used to ingest data into operational pipelines.

- **CI and developer-environment inputs are trusted.**
  Several `subprocess(shell=True)` call sites in CI helpers and dev
  scripts interpolate variables into shell commands. Untrusted PR
  metadata or environment variables should not be allowed to reach
  those scripts; rely on the org-level CI hardening (least-privilege
  workflow permissions, SHA-pinned actions) rather than expecting
  per-script defense.

- **GPU memory is not a confidentiality boundary.**
  Multiple processes sharing a GPU, or co-tenants on a shared host, may
  observe each other's GPU memory through driver-level side channels.
  cuGraph assumes the caller has provisioned the GPU appropriately
  (MIG, exclusive process, container isolation) when confidentiality
  matters.

## Supported Versions

Security fixes are issued against the current release line published on the
[RAPIDS release schedule](https://docs.rapids.ai/releases/). Older minor
releases are generally not backported; upgrade to the latest supported
version to receive fixes.

## Dependency Security

cuGraph tracks CVEs in its direct dependencies — notably `cudf`,
`pylibraft` / `raft-dask`, `ucxx`, `dask-cuda`, `cupy`, `numba`,
`cuda-python`, `fsspec`, and `rmm`. Dependency updates ship with regular
releases; high-severity upstream CVEs may trigger out-of-band patch
releases.
