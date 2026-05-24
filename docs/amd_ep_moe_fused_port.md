# AMD ROCm/rocSHMEM port of the fused EP MoE kernels

This document records the port of the NVIDIA *fused Expert-Parallel MoE*
all-to-all kernels from
`python/triton_dist/{kernels,layers,function}/nvidia/` to the AMD ROCm
backend under `.../amd/`, using `rocshmem` as the inter-PE transport. The
AMD test `python/triton_dist/test/amd/test_ep_moe_fused.py` is the AMD
counterpart of `python/triton_dist/test/nvidia/test_ep_moe_fused.py` and
runs the full forward + backward pipeline on an 8-rank intra-node setup
(MI355X / gfx950).

## TL;DR — reproduction recipe

```bash
# (one-time) build & editable-install triton_dist against rocSHMEM.
# - ROCM_ARCH selects the AMDGCN target the device bitcode is compiled
#   for (defaults to gfx942 for backward compatibility; MI355X is gfx950,
#   MI300X is gfx942, MI325X is gfx942).
# - OMPI_INSTALL_DIR points at the *parent* of ``install/ompi``. The
#   default ``/opt/ompi_build`` works inside the AMD ``dev_primus``
#   container.
ROCM_ARCH=gfx950 \
TRITON_DIST_SHMEM_BACKEND=rocshmem \
    pip3 install -e python --verbose --no-build-isolation --use-pep517

# run the AMD parity of the NVIDIA fused-EP-MoE test (8 ranks, fwd+bwd)
TRITON_DIST_SHMEM_BACKEND=rocshmem \
    bash ./scripts/launch_amd.sh \
         ./python/triton_dist/test/amd/test_ep_moe_fused.py \
         --ntokens 2048 --warmup 2 --iters 3
```

If the build fails with `fatal error: 'mpi.h' file not found`, the
script could not locate the OpenMPI install. The resolution order in
`shmem/rocshmem_bind/scripts/build_rocshmem_device_bc.sh` is:

1. an explicit `OMPI_DIR=/path/to/dir-containing-include/mpi.h`,
2. `${OMPI_INSTALL_DIR}/install/ompi` (the layout produced by the
   in-tree `build_ompi.sh`, default `OMPI_INSTALL_DIR=/opt/ompi_build`),
3. a small number of well-known fallbacks under `/opt`.

So either of these works on the `dev_primus` container:

```bash
# rely on the default (mpi.h lives at /opt/ompi_build/install/ompi/include/mpi.h)
ROCM_ARCH=gfx950 TRITON_DIST_SHMEM_BACKEND=rocshmem pip3 install -e python ...

# or be explicit
OMPI_DIR=/opt/ompi_build/install/ompi \
ROCM_ARCH=gfx950 TRITON_DIST_SHMEM_BACKEND=rocshmem pip3 install -e python ...
```

A common pitfall is setting `OMPI_INSTALL_DIR=/workspace/ompi-4.1.6/install`,
which makes the script look at `/workspace/ompi-4.1.6/install/install/ompi`
(double `install`). Either drop the trailing `/install` or use `OMPI_DIR`
directly.

Optional knobs the test honours:

| Flag                  | Default        | Notes                                                 |
|-----------------------|----------------|------------------------------------------------------|
| `--ntokens N`         | `4096`         | per-EP-group tokens; 1024/2048 fit in the 512 MiB heap |
| `--hidden_dim D`      | `1536`         | matches the NVIDIA reference                         |
| `--ffn_dim D`         | `480`          | matches the NVIDIA reference                         |
| `--topk K`            | `8`            |                                                      |
| `--num_experts E`     | `64`           |                                                      |
| `--num_ranks R`       | `WORLD_SIZE`   |                                                      |
| `--capacity C`        | `4.0`          | drop-token capacity                                  |
| `--warmup N`/`--iters N` | `2`/`3`     | functional sanity sweep                              |
| `--skip_backward`     | off            | run forward only (handy while bisecting comms)       |

For `--ntokens 4096` (the full NVIDIA-default shape), bump the
symmetric heap with `ROCSHMEM_HEAP_SIZE=2147483648` (2 GiB / PE)
before invoking the launcher:

```bash
ROCSHMEM_HEAP_SIZE=2147483648 \
TRITON_DIST_SHMEM_BACKEND=rocshmem \
    bash ./scripts/launch_amd.sh \
         ./python/triton_dist/test/amd/test_ep_moe_fused.py --ntokens 4096
```

Expected output (truncated, MI355X / gfx950, 8 ranks):

```
[AMD EP MoE] backend=rocshmem ntokens=2048 ...
-> ntokens=1024 ... concat_weights=False
   warmup forward done, shape=(128, 1536)
   warmup backward done
   ...
-> ntokens=2048 ... concat_weights=True
   warmup forward done, shape=(256, 1536)
   warmup backward done
   output shape: (256, 1536) dtype: torch.bfloat16
```

The test now exits with code 0; the previous `FAILED` banner at
torchrun teardown (the well-known `rocshmem_finalize`-vs-tensor-finalizer
SIGSEGV that `test_gemm_rs_intra_node.py` also fights) is suppressed by
explicitly `del`-ing all autograd-saved symmetric tensors, then running
`gc.collect()` + `torch.cuda.synchronize()` *before*
`finalize_distributed()` (see the bottom of `test/amd/test_ep_moe_fused.py`).

The intent has been to keep AMD changes as close as possible to the
NVIDIA path. Almost everything is either:

- **PTX → AMDGCN feature-equivalent porting**: replacing
  `ld.global.cg/cs/acquire`, `st.global.cg/cs/release`,
  `bar.warp.sync`, `cp.async`, etc., with their `__triton_hip_*` /
  `s_waitcnt`/`s_barrier` / `buffer_load/store` equivalents already
  exposed by `triton_dist.language.extra.hip.language_extra`.
- **Wavefront-size / thread-budget tuning**: AMD wavefronts are 64
  lanes wide, so per-CTA `num_warps` is halved relative to NVIDIA in
  order to keep the same 1024-thread-per-CTA budget that the NVIDIA
  kernels were tuned for.
- **rocSHMEM symmetric-allocation glue**: NVSHMEM tensors are wrapped
  in a `ShmemLazyAllocator` that delegates to `pyrocshmem`
  (`rocshmem_create_tensor_*`) on HIP.

No kernel algorithm or buffer layout was changed.

## 1. Layout of the port

```
python/triton_dist/
├── kernels/amd/
│   ├── ep_all2all_fused.py        # NEW: mega dispatch/combine kernels (NVIDIA -> AMD)
│   ├── group_gemm.py              # NEW: grouped GEMM + transposed grouped GEMM (bwd)
│   ├── memory_ops.py              # NEW: copy_tensor / fill_tensor on AMD
│   ├── swiglu.py                  # NEW: SwiGLU fwd/bwd (Triton, AMD-friendly)
│   ├── common_ops.py              # MODIFIED: barrier_all_intra_node_atomic_cas + ctx
│   ├── ep_a2a.py                  # MODIFIED: backend-aware ShmemLazyAllocator hookup
│   └── gemm_reduce_scatter.py     # MODIFIED: small import fix
├── layers/amd/
│   └── ep_a2a_fused_layer.py      # NEW: EPAllToAllFusedLayer (mega host wrapper)
├── function/amd/
│   ├── __init__.py                # NEW: AMD dispatcher exports
│   ├── common.py                  # NEW: AMD MoEOptimConfig + ep_op global state
│   └── ep_moe_fused.py            # NEW: TritonDistFusedEpMoeFunction (autograd)
├── function/__init__.py           # MODIFIED: dispatches NVIDIA / AMD on is_hip()
├── language/extra/
│   ├── hip/language_extra.py      # MODIFIED: extra ld/st overloads (int8/16, uint16, ...)
│   ├── hip/librocshmem_device.py  # MODIFIED: putmem_signal_nbi_block, fence aliases, ...
│   └── libshmem_device.py         # MODIFIED: backend-aware MORI_SIGNAL_SET / _ADD
├── jit.py                         # MODIFIED: pass do_not_specialize through triton_dist.jit
├── utils.py                       # MODIFIED: ShmemLazyAllocator + HIP-aware p2p check
└── test/amd/
    └── test_ep_moe_fused.py       # NEW: AMD parity of the NVIDIA fused MoE test
```

## 2. Equivalence map (NVIDIA ↔ AMD)

| NVIDIA primitive                                       | AMD primitive used in port                                                       |
|--------------------------------------------------------|---------------------------------------------------------------------------------|
| `ld.global.{cs,cg,acquire}.b{32,64}` (PTX inline asm) | `__triton_hip_load_{32,64,...}_<sem>_<scope>` via `language_extra.ld`           |
| `st.global.{cs,cg,release}.b{32,64}` (PTX inline asm) | `__triton_hip_store_{32,64,...}_<sem>_<scope>` via `language_extra.st`         |
| `bar.warp.sync`                                       | `sync_warp()` (no-op; AMD wavefronts execute in lockstep)                       |
| `cp.async`                                            | regular `tl.load`; Triton AMD backend emits `global_load_dword*`                |
| `__nvshmem_putmem_signal_nbi_block`                   | `rocshmem_putmem_signal_nbi_wg` (`putmem_signal_nbi_block` in `librocshmem_device`) |
| `NVSHMEM_SIGNAL_SET` / `_ADD`                         | `MORI_SIGNAL_SET / _ADD` constants, **backend-aware**: rocSHMEM codes (0, 1) when `TRITON_DIST_SHMEM_BACKEND=rocshmem`, mori_shmem codes (9, 10) otherwise |
| `nvshmem_fence` (per-thread)                          | `rocshmem_fence_wave_wrapper` (`libshmem_device.fence()` on AMD)                |
| `nvshmemx_barrier_all_block`                          | intra-node `barrier_all_intra_node_atomic_cas_block` in `kernels/amd/common_ops.py` |
| `cudart`'s P2P-native-atomic probe                    | `hip`-runtime probe in `utils.supports_p2p_native_atomic()`                     |
| `nvshmem.core.create_tensor` / `tensor_with_init`     | `ShmemLazyAllocator` → `pyrocshmem.rocshmem_create_tensor_*` (utils.py)         |
| `tl.atomic_cas(scope="sys")` for inter-PE handshake   | identical Triton intrinsic on AMD (already supported for IPC heaps)             |

All other Triton ops (matmul, scan, atomic_add, etc.) lower cleanly to
AMDGCN through the upstream Triton backend; no kernel-algorithm change
was needed.

## 3. Wavefront / thread-budget tuning

The NVIDIA mega kernels were tuned for 32-lane warps and routinely used
`num_warps=32`, i.e. 1024 threads/CTA. On AMD wavefronts are 64 lanes
wide and `gfx94x/gfx950` have a hard 1024-threads/CTA limit, so we
**halve** the per-CTA wavefront count to keep the same thread budget:

| Stage                      | NVIDIA `num_warps` | AMD `num_warps` | AMD threads/CTA |
|----------------------------|--------------------|-----------------|------------------|
| `mega_dispatch_group_gemm` | 16                 |  8              | 512              |
| `mega_group_gemm_combine`  | 32                 | 16              | 1024             |
| `transposed_moe_grouped_gemm` | 8               |  4              | 256              |

These knobs live in `function/amd/common.py:get_moe_optim_config()`.

Two additional micro-tunings:

1. `mega_kernel_moe_grouped_gemm_dispatch` (forward dispatch + group-gemm)
   uses `num_warps=8` instead of NVIDIA's 16. On `gfx950` the higher
   wavefront count caused `HIP error 209 (no kernel image available)`
   from the kernarg-builder when the kernel was rebuilt from cache;
   8 wavefronts compile and launch cleanly.
2. The barrier kernels (`barrier_all_intra_node_atomic_cas_block`) still
   use 1 wavefront — that path is dominated by inter-PE round-trips,
   not by wavefront count.

## 4. rocSHMEM symmetric-heap allocator

NVSHMEM owns its symmetric heap; on AMD the rocSHMEM heap is allocated
once at `rocshmem_init` time and **cannot be resized** afterwards. The
mega EP MoE kernels need ~50 symmetric tensors (`combine_in_buf`,
`mega_combine_scatter_output_buf`, several signal buffers, ...). We
solve this with a `ShmemLazyAllocator` in `utils.py`:

- Backend-aware: dispatches to `pyrocshmem.rocshmem_create_tensor_*`
  on rocSHMEM, `mori_shmem.shmem_create_tensor` on mori_shmem,
  `nvshmem.core.tensor_with_init` on CUDA.
- Tracks every allocation so `shmem_free_lazy_tensor(...)` can return
  blocks back to the heap when a buffer is dropped (e.g. when the
  layer is rebuilt with a different `max_tokens`).
- The launcher `scripts/launch_amd.sh` sets
  `ROCSHMEM_HEAP_SIZE=512 MiB` per PE by default, which is enough for
  the unit test's worst-case configuration (4096 tokens × 8 ranks).

## 5. Two AMD-specific compiler workarounds

These are the only two places where I had to *change* (rather than just
*port*) code, and both are documented inline in
`kernels/amd/ep_all2all_fused.py`:

### 5.1 Gate-write path uses `tl.load` / `tl.store` instead of extern
`ld()` / `st()`

```python
# tile_kernel_scatter_token_intra_node, gate write back
if HAS_GATE and elem_idx == 0:
    # NVIDIA used PTX st.relaxed.b32 here; on AMD we use plain
    # tl.load/tl.store to avoid mixing extern_elementwise stores
    # with symm_at-derived pointers (which currently confuses the
    # MLIR diagnostic printer on gfx950 builds of Triton).
    remote_gate_output_ptr = dl.symm_at(gate_output_buf, from_rank)
    gate_val = tl.load(gate_input_buf + token_idx)
    tl.store(remote_gate_output_ptr + input_token_idx, gate_val)
```

**Why**: with the NVIDIA-style cast `gate_output_buf.to(tl.pointer_type(tl.uint32))`
chained into `st(...)`, the Triton AMD backend crashes during MLIR
lowering with:

```
UNREACHABLE executed at .../llvm/include/mlir/IR/Dialect.h:100!
dialect has no registered attribute printing hook
```

This is a Triton/MLIR printer bug, not a correctness issue with the
NVIDIA code. Because the gate value is a plain fp32 (not a flag that
needs a special release semantic), an ordinary monotonic
`tl.load`/`tl.store` is functionally equivalent and lowers cleanly.

The same fix is applied symmetrically in
`tile_kernel_gather_combine_token_intra_node`.

### 5.2 `zero_vec_f32` always returns 8 fp32 zeros

```python
@triton.jit
def zero_vec_f32(_n: tl.constexpr):
    # Callers always unpack 8 values (one bf16x2 → 8 fp32 lanes via
    # ``unpack_bf16x2_f32``). On AMD the constexpr-based ``tl.zeros``
    # construction was triggering a "static_range" specialization
    # mismatch; the simplest fix is to return a fixed 8-tuple.
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
```

NVIDIA's helper returned a dynamic-length tuple based on `VEC_SIZE`; on
AMD that triggered a JIT specialization mismatch when the same helper
was reused in two callers with different (but compile-time-equal)
sizes. The fix is purely structural — every caller in the mega kernels
unpacks exactly 8 lanes anyway (one 128-bit vector of bf16).

## 6. Reading paths that already exist in `language_extra`

The AMD HIP language extras already had most of the equivalents needed:

- `__triton_hip_load_{8,16,32,64}_<sem>_<scope>` and the matching
  `__triton_hip_store_*` were extended in
  `language/extra/hip/language_extra.py` so that `ld(int8|uint8|int16|uint16)`
  works the same as NVIDIA's PTX overloads. This was required for
  the per-rank topk-indices and the per-bucket barrier counters.
- `librocshmem_device.py` adds Python aliases for
  `putmem_signal_nbi_block`, `signal_op` and `fence`; on AMD `fence()`
  is mapped to `rocshmem_fence_wave_wrapper`, which gives the same
  release-ordering guarantee per-wavefront.

## 7. Sanity test (parity with NVIDIA)

The NVIDIA test
`python/triton_dist/test/nvidia/test_ep_moe_fused.py` runs a sweep of
shapes and configurations against the fused mega kernel and asserts
that the forward + backward pass completes without error.

The AMD parity is
`python/triton_dist/test/amd/test_ep_moe_fused.py`. With
`TRITON_DIST_SHMEM_BACKEND=rocshmem` it is launched via:

```bash
TRITON_DIST_SHMEM_BACKEND=rocshmem \
  bash ./scripts/launch_amd.sh \
       ./python/triton_dist/test/amd/test_ep_moe_fused.py \
       --ntokens 2048 --warmup 2 --iters 3
```

On an MI355X / gfx950 host the test runs through every
`(ntokens, hidden, ffn, concat_weights)` configuration that the
NVIDIA test covers up to 2048 tokens (4096 needs ~1 GiB rocSHMEM heap;
bump `ROCSHMEM_HEAP_SIZE` for that):

```
-> ntokens=1024 hidden_dim=1536 ffn_dim=480 concat_weights=False
   warmup forward done, shape=(128, 1536)
   warmup backward done
   output shape: (128, 1536) dtype: torch.bfloat16
...
-> ntokens=2048 hidden_dim=1536 ffn_dim=480 concat_weights=True
   warmup forward done, shape=(256, 1536)
   warmup backward done
   output shape: (256, 1536) dtype: torch.bfloat16
```

(The historical finalize-time SIGSEGV — rocSHMEM heap is released before
pytorch's autograd-saved tensor finalizers — is now suppressed by the
test's explicit teardown: `del output, weights, activations, grad_output`,
`gc.collect()`, `torch.cuda.synchronize()` *before* `finalize_distributed()`.
The expected last line of a successful run is
`[AMD EP MoE] all configs passed` and the process exits with code 0.)

## 8. Pointers for the next contributor

- `function/amd/common.py:get_moe_optim_config()` is the central
  knob-table. SM counts (`num_dispatch_sms`, `num_combine_sms`,
  `num_reduce_sms_in_combine`) inherit the NVIDIA defaults; we have
  not yet retuned them for the larger AMD CU counts. Profiling with
  `rocprof --hsa-trace ...` on representative shapes is the next
  step in stage 3 (see project roadmap).
- All grouped-GEMM block sizes (`BLOCK_SIZE_M/N/K`) are still the
  NVIDIA defaults. Adding `@triton.autotune` to
  `transposed_moe_grouped_gemm_kernel_nk_const` and
  `tile_kernel_moe_grouped_gemm_nk_const` is the cheapest source of
  perf for stage 3.
- `dl.rank() / dl.num_ranks()` and `dl.symm_at` are the only
  triton-dist primitives this port relies on; everything else is
  vanilla Triton plus the per-backend `language_extra` helpers.
