# Code Review: AMD Port of Triton-distributed

**Branch:** `dev/port-amd`
**Commits reviewed:** `c250ad0..575371c` (3 commits ahead of `origin/main`)

- `575371c` — amd: reorder barrier_all_on_stream signature
- `4491131` — function: add amd stub package
- `c250ad0` — port to amd

## Methodology

Five parallel reviewers covered: shallow bug scan, NVIDIA/AMD parity check, git-history context, code-comment compliance, and HIP/ROCm correctness. Findings were then verified against the actual source — many initially-flagged issues turned out to be faithful ports of pre-existing NVIDIA patterns and were filtered out. Specifically, the following are **not regressions** introduced here (they exist on `origin/main` too):

- `barrier_all_on_stream` not propagating `stream=` into the intra-node Triton kernel launch
- `BarrierAllContext.symm_barrier` allocated as `(num_local_ranks,)` while `barrier_all_intra_node_non_atomic_block` docstring requires `2*num_local_ranks` slots
- No `__syncthreads()` between phase-1 and phase-2 of `barrier_all_intra_node_atomic_cas_block`

Reference: `python/triton_dist/kernels/nvidia/common_ops.py:154-242`.

---

## Real issues introduced by this branch

### 1. `scripts/launch_amd.sh` — rocSHMEM/MPI `LD_LIBRARY_PATH` export disabled

Commit `c250ad0` comments out the entire backend-gated `LD_LIBRARY_PATH` export block:

```bash
# if [ "${TRITON_DIST_SHMEM_BACKEND}" != "mori_shmem" ]; then
#   export LD_LIBRARY_PATH=${MPI_ROOT}/lib:${ROCSHMEM_ROOT}/lib:$LD_LIBRARY_PATH
# fi
```

The default `ROCSHMEM_BACKEND=IPC` path (set in the same file at line 26) will now fail to `dlopen` `librocshmem.so` / `libmpi.so` unless the container image already places them on the default loader search path. The guard's purpose — to exclude the `mori_shmem` backend — was the only thing protecting the rocSHMEM-backend users this PR is otherwise extending.

**Location:** `scripts/launch_amd.sh:11-13`

**Suggested fix:** Restore the original guarded export, or document the new container-image preconditions.

---

### 2. MPI install path divergence across the rocshmem build chain

`build_rocshmem.sh` now hard-codes `MPI_ROOT=/opt/ompi` and `UCX_ROOT=/opt/ucx`, and `shmem/rocshmem_bind/pyrocshmem/setup.py` was updated to match. But the orchestrator `shmem/rocshmem_bind/build.sh` (unchanged) still computes:

```bash
export OPENMPI_UCX_INSTALL_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"
export PATH="${OPENMPI_UCX_INSTALL_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${OPENMPI_UCX_INSTALL_DIR}/lib:$LD_LIBRARY_PATH"
# ...and passes -DOMPI_DIR=${OPENMPI_UCX_INSTALL_DIR} to pyrocshmem cmake
```

Before this PR all three paths agreed (`/opt/ompi_build/install/ompi`). Afterwards, `build.sh` feeds `/opt/ompi_build/install/ompi` into PATH/LD/cmake while the child build script and `pyrocshmem/setup.py` look at `/opt/ompi`.

**Location:** `shmem/rocshmem_bind/build_rocshmem.sh:60-78` vs `shmem/rocshmem_bind/build.sh:50-55`

**Suggested fix:** Either update `build.sh` to use the new convention (`MPI_ROOT=/opt/ompi`), or restore the env-overridable `OMPI_INSTALL_DIR` pattern in `build_rocshmem.sh` so all three layers stay in sync.

---

### 3. `rocshmem_free_tensor_sync` does not actually free

```python
def rocshmem_free_tensor_sync(tensor):
    """rocshmem symmetric tensors are freed when the Python ``SymmRocShmemBuffer``
    is garbage-collected. We synchronize so pending GPU work is drained
    *before* the caller drops its last reference, matching the semantics of
    :func:`nvshmem_free_tensor_sync`.
    """
    torch.cuda.synchronize()
```

The sibling `nvshmem_free_tensor_sync` calls `nvshmem.core.free_tensor(tensor)` between two synchronizes; `mori_shmem_free_tensor_sync` calls `mori_shmem.mori_shmem_free_tensor(tensor)`. The new rocshmem variant accepts the `tensor` argument and drops it on the floor. The docstring claim "matching the semantics of `nvshmem_free_tensor_sync`" is false — it relies on Python GC to invoke the `SymmRocShmemBuffer` destructor, which won't happen if any caller still holds a reference (e.g. via the new `ShmemLazyAllocator` bookkeeping). The symmetric heap will leak across `BarrierAllContext.finalize()` calls and the new EP-MoE stubs once they wire up.

**Location:** `python/triton_dist/utils.py:326-332`

**Suggested fix:** Either invoke the real free entry point in `pyrocshmem`, or change the docstring + raise on use until the freeing API is wired.

---

### 4. `BarrierAllContext.local_rank` derivation unsafe under non-contiguous PE layout

```python
self.rank = get_triton_dist_world().rank()
self.local_world_size = (
    get_triton_dist_local_world_size()
    or int(_os.environ.get("LOCAL_WORLD_SIZE", "0"))
    or int(_os.environ.get("WORLD_SIZE", "1"))
)
self.local_rank = self.rank % self.local_world_size
```

NVIDIA derives `local_rank` from `pynvshmem.team_my_pe(TEAM_NODE)` — the authoritative position in the SHMEM node team. The `rank % local_world_size` substitute is only correct when global PE ids are contiguous within each node and start at PE 0 per node. The `local_rank_offset = rank - local_rank` arithmetic in `barrier_all_intra_node_atomic_cas_block` (lines 213-219) will then address the wrong peer flag slot under any other layout — silently, not loudly.

Worse: the chained `or int(_os.environ.get("WORLD_SIZE", "1"))` fallback fires on multi-node runs where `LOCAL_WORLD_SIZE` is not set, returning `local_world_size = WORLD_SIZE` (e.g. 16 for 2×8). The follow-on `symm_barrier` is allocated as `(num_local_ranks,)`, so the intra-node CAS kernel runs with an oversized index space against an undersized symmetric buffer → out-of-bounds access on the symmetric heap.

**Location:** `python/triton_dist/kernels/amd/common_ops.py:281-295`

**Suggested fix:** Gate `BarrierAllContext` on a rocSHMEM/mori node-team query if available; otherwise raise when neither `get_triton_dist_local_world_size()` nor `LOCAL_WORLD_SIZE` is set rather than silently using `WORLD_SIZE`.

---

### 5. `NVSHMEM_SIGNAL_DTYPE` shadowed as `uint64` in AMD module

```python
# python/triton_dist/kernels/amd/common_ops.py
NVSHMEM_SIGNAL_DTYPE = MORI_SHMEM_SIGNAL_DTYPE  # = torch.uint64
```

The canonical `triton_dist.utils.NVSHMEM_SIGNAL_DTYPE` is `torch.int64` (`python/triton_dist/utils.py:661`), and NVIDIA code at `python/triton_dist/kernels/nvidia/common_ops.py:375,397` uses it for `signal_tensor.dtype` dispatch. The AMD module rebinds the same name to `torch.uint64` at module scope, justified by a comment as preserving import compatibility for "downstream layer code". But anything that does `from triton_dist.kernels.amd.common_ops import NVSHMEM_SIGNAL_DTYPE` and compares against `signal_tensor.dtype` (the NVIDIA pattern) or `.view(torch.int64)` will silently take the wrong branch.

**Location:** `python/triton_dist/kernels/amd/common_ops.py:55-60`

**Suggested fix:** Set the alias to `torch.int64`, or do an explicit rename rather than shadowing the canonical symbol.

---

## Worth a look (lower confidence)

### A. `get_moe_optim_config` not actually verbatim

`python/triton_dist/function/amd/common.py` is described in the commit message for `4491131` as "ported verbatim", but `get_moe_optim_config` adds `min(80, max_sms)` and `min(64, max_sms)` clamps that NVIDIA doesn't have, and drops the `max_sms > 78` H800/H20 split. On every current MI300/MI355 SKU (CU count < 80/64 in some partitioned modes), the AMD path will silently downsize `num_dispatch_sms` / `num_combine_sms`. Confirm intent — if intentional, drop the "verbatim" claim.

### B. `MORI_SIGNAL_SET` resolved at import time

`python/triton_dist/language/extra/libshmem_device.py:557-563` switches `MORI_SIGNAL_SET` between `0` (rocSHMEM) and `9` (mori) by reading `TRITON_DIST_SHMEM_BACKEND` once at import. If anything imports `libshmem_device` before that env var is set, the cached constant is wrong and the matching `wait_until` deadlocks — and `scripts/launch_amd.sh` does not currently export `TRITON_DIST_SHMEM_BACKEND`. Consider asserting backend at first use or resolving lazily.

### C. `fence()` routed to `rocshmem_fence_wave_wrapper`

`python/triton_dist/language/extra/hip/librocshmem_device.py:433-452` redirects the public `fence()` extern to the `_wave` wrapper because the non-suffixed `rocshmem_fence_wrapper` symbol is absent from the prebuilt bitcode. The inline comment claims they are "functionally equivalent" because internally the wrapper just calls `rocshmem::rocshmem_fence()`. Worth sanity-checking against rocSHMEM's collective-precondition documentation since the `_wave` suffix usually implies a wavefront-collective requirement; callers like `low_latency_all_to_all.py` invoke `fence()` from a single thread.

---

## Verified clean / intentional

- `barrier_all_on_stream` signature reorder in `575371c` correctly restores positional `stream` for legacy callers (`test_distributed-notify-wait.py`) and simplifies the two `gemm_reduce_scatter` call sites.
- `kernels/amd/ep_a2a.py kernel_get_dispatch_send_reqs` body is byte-equivalent to the NVIDIA version.
- `jit.py` changes are gated on `TRITON_DIST_DEBUG_ROCSHMEM_CTX` and have no effect on the NVIDIA path.
- `setup.py` `ROCM_ARCH` change is env-overridable with `gfx942` default; NVIDIA build path is untouched.
- The new `language_extra.py` int8/uint8/int16 overloads for `ld`/`st` are additive and match the NVIDIA surface.
- The `_block` / `_warp` aliases in `librocshmem_device.py` correctly mirror the NVSHMEM naming convention onto rocSHMEM's `_wg` / `_wave` primitives.
- `function/amd/__init__.py` and `function/amd/ep_moe_fused.py` correctly raise `NotImplementedError` as stubs.

---

## Summary

Five real issues, all of which can be fixed surgically without changing the broader port shape. The build/launch script issues (#1, #2) are the most likely to bite immediately — they will surface as `dlopen` or build failures on the first run from a clean container. The `BarrierAllContext` (#4) and `NVSHMEM_SIGNAL_DTYPE` (#5) issues are silent-corruption hazards that only matter once the AMD fused EP-MoE backend lands and starts exercising these paths. `rocshmem_free_tensor_sync` (#3) is a heap-leak that gets worse the longer the process runs.
