# Security Policy

## Reporting Security Issues

NVIDIA takes security vulnerabilities seriously. If you believe you have found a security vulnerability in cuGraph, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.**

Instead, please send an email to `security@nvidia.com` with the following information:
- Type of issue (e.g., buffer overflow, SQL injection, race condition, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

## ATGC (Adversarial Topology Against GPU Compute) Vulnerability

### CVE ID
- **CVE-2026-XXXX** (pending assignment)

### Description
A vulnerability class exists in GPU graph analytics frameworks where adversarial graph topologies (specifically "clique-chain" graphs) can cause exponential per-thread work, leading to GPU kernel hangs and TDR (Timeout Detection and Recovery) timeouts.

### Impact
- **Severity**: High (CVSS 8.2)
- **CWE**: CWE-400 (Uncontrolled Resource Consumption)
- **Affected**: All GPU graph frameworks using per-thread frontier algorithms
- **Specific Impact**: Denial of Service on GPU resources

### Mitigation
This version of cuGraph includes an ATGC detection module that:
1. Identifies adversarial clique-chain topologies at graph ingestion time
2. Blocks or warns about suspicious inputs before GPU dispatch
3. Provides configurable security policies via environment variables

### Configuration
- `CUGRAPH_ENABLE_ATGC_GUARD`: Enable/disable ATGC detection (default: `true`)
- `CUGRAPH_ATGC_ACTION`: Action on detection - `reject`, `warn`, or `allow` (default: `reject`)
- `CUGRAPH_ATGC_LOG_LEVEL`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR` (default: `WARNING`)

### Timeline
- **Discovery**: June 9, 2026
- **Responsible Disclosure**: Pending (NVIDIA PSIRT)
- **Patch Release**: TBD
- **Public Disclosure**: TBD (90-day embargo)

## Security Best Practices

When deploying cuGraph in production environments:

1. **Enable ATGC Guard**: Keep `CUGRAPH_ENABLE_ATGC_GUARD=true` in production
2. **Monitor Logs**: Watch for ATGC warnings in application logs
3. **Input Validation**: Validate graph inputs before ingestion when possible
4. **Resource Limits**: Set appropriate GPU timeout limits
5. **Regular Updates**: Keep cuGraph updated to latest security patches

## Security Updates

Security updates will be released as part of regular RAPIDS releases. Critical security patches may be released out-of-band.

Subscribe to the [RAPIDS Security Advisories](https://docs.rapids.ai/notices) for security updates.

## Acknowledgments

We thank the security researchers who responsibly disclosed this vulnerability:
- **Ankit Chetri** (`ankit.byte.404@gmail.com`) — Pipeline & Testing
- **Teerth Sharma** (`teerths57@gmail.com`) — Topology & Defensive Analysis

## References

- NVIDIA CUDA TDR Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#timeout-detection-and-recovery
- CWE-400: https://cwe.mitre.org/data/definitions/400.html
- CVSS v3.1 Calculator: https://www.first.org/cvss/calculator/3.1
