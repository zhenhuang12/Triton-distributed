#!/bin/bash
# Launcher for the rocSHMEM IPC bandwidth test.
# Run inside the primus pr-715-ainic container on smci355 (gfx950).
set -uo pipefail

DIST_DIR=/apps/zhuang12/MegaKernel/Triton-distributed
cd "${DIST_DIR}"

# memlock must be unlimited so rocshmem can pin the symmetric heap
ulimit -l unlimited

# Pre-set the heap (in-tree rocshmem reads it at rocshmem_init)
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-1073741824}

# Force the in-tree rocshmem IPC backend (prebuilt /opt/rocshmem is GDA-only)
unset ROCSHMEM_HOME
export ROCSHMEM_BACKEND=IPC

# Pick up the in-tree pyrocshmem + triton_dist
export PYTHONPATH="${DIST_DIR}/python:${DIST_DIR}/shmem/rocshmem_bind/pyrocshmem/build/lib.linux-x86_64-cpython-312:${DIST_DIR}/shmem/rocshmem_bind/python:${PYTHONPATH:-}"

export LD_LIBRARY_PATH="${DIST_DIR}/shmem/rocshmem_bind/rocshmem_build/install/lib:${LD_LIBRARY_PATH:-}"

# Use the project launcher (sets the rest of the env vars + torchrun args)
exec bash scripts/launch_amd.sh \
  python/triton_dist/test/amd/test_bandwidth.py \
  --check \
  "$@"
