#!/bin/bash
###############################################################################
# AMD mega EP-MoE smoke run inside dev_primus container
# (driver for python/triton_dist/test/amd/test_ep_moe_fused.py)
###############################################################################
set -euo pipefail

cd "$(dirname "$0")"

ROOT=$(pwd)
ROCSHMEM_INSTALL=${ROOT}/shmem/rocshmem_bind/rocshmem_build/install
PYROCSHMEM_LIB=${ROOT}/shmem/rocshmem_bind/pyrocshmem/build/lib.linux-x86_64-cpython-312

# Point at the in-tree rocSHMEM build (IPC+GDA+RO), not /opt/rocshmem (GDA-only).
export ROCSHMEM_HOME=${ROCSHMEM_INSTALL}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:-IPC}
# rocSHMEM heap is pinned at init; default 512 MiB only fits --ntokens<=1024.
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}  # 8 GiB
# mori_shmem default static heap (4 GiB) is too small for ntokens>=1024 on this
# test (~7.2 GiB required). Same 8 GiB number works for both backends.
export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-8589934592}
export MORI_SHMEM_SYMMETRIC_SIZE=${MORI_SHMEM_SYMMETRIC_SIZE:-8589934592}
export TRITON_DIST_SHMEM_BACKEND=${TRITON_DIST_SHMEM_BACKEND:-rocshmem}
# rocSHMEM TCP bootstrap defaults to a 5-second accept window; on a cold
# torchrun the other ranks can take longer than that just to import torch.
export ROCSHMEM_BOOTSTRAP_TIMEOUT=${ROCSHMEM_BOOTSTRAP_TIMEOUT:-120}
# Force the bootstrap to use loopback — with --network=host all 8 ranks
# share the namespace and lo is always reachable; the auto-pick path
# can choose a benic/fenic NIC whose connectivity from inside the
# container is harder to predict.
export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME:-lo}

# Make the pre-built artifacts importable without re-running pip install:
#   triton_dist  -> python/
#   pyrocshmem   -> shmem/rocshmem_bind/pyrocshmem/build/lib.../pyrocshmem
export PYTHONPATH=${ROOT}/python:${PYROCSHMEM_LIB}:${PYTHONPATH:-}

# rocSHMEM pins its symmetric heap; container default memlock (8 MiB) is too small.
ulimit -l unlimited

NTOKENS=${NTOKENS:-8192}  # matches test_ep_moe_fused.py default; sweeps 1024..NTOKENS
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-15}

exec bash ./scripts/launch_amd.sh \
    ./python/triton_dist/test/amd/test_ep_moe_fused.py \
    --ntokens "${NTOKENS}" --warmup "${WARMUP}" --iters "${ITERS}" "$@"
