#!/usr/bin/env bash
# Read-only prerequisite check. Does not install, restart or change drivers.
set -u
status=0

printf 'Checking local CUDA prerequisites\n'
cuda_compiler=${NVCC:-nvcc}
if command -v "$cuda_compiler" >/dev/null 2>&1; then
    if ! "$cuda_compiler" --version; then status=1; fi
else
    printf 'MISSING: CUDA compiler (%s). The toolkit is required.\n' "$cuda_compiler" >&2
    status=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
        printf 'WARNING: NVIDIA monitoring failed. The application will test CUDA directly and stop if computation is unavailable.\n' >&2
    fi
else
    printf 'WARNING: nvidia-smi is missing. The application will test CUDA directly.\n' >&2
fi

for dependency in g++ make python3; do
    if ! command -v "$dependency" >/dev/null 2>&1; then
        printf 'MISSING: %s\n' "$dependency" >&2
        status=1
    fi
done
if [ "$status" -eq 0 ]; then
    printf 'Compiler prerequisites found. Compilation and the application will check CUDA compatibility and execution.\n'
fi
exit "$status"
