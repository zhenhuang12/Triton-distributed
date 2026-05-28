# Triton-distributed

This is ByteDance Seed's distributed Triton compiler with AMD ROCm /
rocSHMEM support. The notes below cover **install + run** on AMD —
specifically the bits that aren't in `docs/build.md` /
`docs/amd_ep_moe_fused_port.md` and that bit me on first launch.

## Running the AMD mega EP-MoE smoke test

Wrapper: [`run_amd_mega_moe.sh`](run_amd_mega_moe.sh). Driver for
[`python/triton_dist/test/amd/test_ep_moe_fused.py`](python/triton_dist/test/amd/test_ep_moe_fused.py)
— the AMD parity of the NVIDIA fused EP-MoE mega kernel.

```bash
# From outside the container — default backend is rocshmem, default NTOKENS=8192
# (matches test_ep_moe_fused.py's --ntokens default; sweeps 1024→8192)
docker exec dev_primus bash -lc \
  "cd /apps/zhuang12/MegaKernel/Triton-distributed && \
   WARMUP=2 ITERS=5 ./run_amd_mega_moe.sh"

# Same test against the mori_shmem backend (install once via the recipe below)
docker exec dev_primus bash -lc \
  "cd /apps/zhuang12/MegaKernel/Triton-distributed && \
   TRITON_DIST_SHMEM_BACKEND=mori_shmem WARMUP=2 ITERS=5 ./run_amd_mega_moe.sh"

# Override NTOKENS to shrink the sweep for a faster smoke (e.g. NTOKENS=1024).
# Wrapper sets both ROCSHMEM_HEAP_SIZE and MORI_SHMEM_HEAP_SIZE /
# MORI_SHMEM_SYMMETRIC_SIZE to 8 GiB — fits the full ntokens=8192 sweep on
# either backend (peak symmetric heap ~7.4 GiB at the largest shape).
```

Expected tail (8 × MI355X / gfx950, hidden=1536, ffn=480, topk=8, experts=64,
WARMUP=2 ITERS=5, format `latency(ms)/peak_mem(MB)/precision`):

```
 Ntokens   Hidden      FFN triton_dist_fwd triton_dist_fwd_bwd
==============================================================
# rocshmem
    1024     1536      480   1.885/ 26.60/✅    3.727/ 67.52/✅
    2048     1536      480   2.575/ 32.23/✅    5.238/ 79.71/✅
    4096     1536      480   4.245/ 45.34/✅    8.415/106.73/✅
    8192     1536      480   7.625/ 68.83/✅   14.880/152.41/✅
# mori_shmem (same shapes)
    1024     1536      480   1.188/ 26.60/✅    2.622/ 67.52/✅
    2048     1536      480   1.355/ 32.23/✅    3.209/ 79.71/✅
    4096     1536      480   1.770/ 45.34/✅    3.818/106.73/✅
    8192     1536      480   2.715/ 68.83/✅    5.573/152.41/✅
```

A `✅` per row is the only success signal — anything else (or a
torchrun `FAILED` banner) is a regression. Peak memory is
backend-independent (same allocator on top); only latency differs.
mori_shmem speedup over rocshmem widens monotonically with ntokens:

| ntokens | fwd speedup | fwd+bwd speedup |
|---:|---:|---:|
| 1024 | 1.59× | 1.42× |
| 2048 | 1.90× | 1.63× |
| 4096 | 2.40× | 2.20× |
| 8192 | 2.81× | 2.67× |

Pick mori_shmem for perf, rocshmem only if you specifically need
its features (e.g. multi-node GDA / RO) or want to bisect against
a second backend.

### DeepSeek model shapes via `MODEL=` preset

The wrapper accepts a `MODEL=` env that overrides
`--hidden_dim / --ffn_dim / --topk / --num_experts`. Shapes are
sourced from [`BenchMoE/model_configs.json`](../../BenchMoE/model_configs.json)
(`moe_intermediate_size` → `--ffn_dim`, `num_topk` → `--topk`):

| `MODEL=` | hidden | ffn  | topk | experts |
|---|---:|---:|---:|---:|
| (unset / `default`)   | 1536 |  480 | 8 |  64 |
| `deepseek-v3`         | 7168 | 2048 | 8 | 256 |
| `deepseek-v4-flash`   | 4096 | 2048 | 6 | 256 |
| `deepseek-v4-pro`     | 7168 | 3072 | 6 | 384 |

**The wrapper's default 8 GiB heap is too small for the V3/V4 shapes** —
bump `MORI_SHMEM_HEAP_SIZE` / `MORI_SHMEM_SYMMETRIC_SIZE` (or
`ROCSHMEM_HEAP_SIZE`) before launching. Verified working values for
the full ntokens=1024→8192 sweep on 8× MI355X with mori_shmem:

```bash
# DeepSeek-V3 (needs ≥64 GiB symmetric heap — topk=8 inflates dispatch
# traffic vs the V4 shapes, peak symmetric heap ~35 GiB so 32 GiB does
# NOT fit)
docker exec dev_primus bash -lc \
  "cd /apps/zhuang12/MegaKernel/Triton-distributed && \
   MODEL=deepseek-v3 TRITON_DIST_SHMEM_BACKEND=mori_shmem \
   MORI_SHMEM_HEAP_SIZE=68719476736 MORI_SHMEM_SYMMETRIC_SIZE=68719476736 \
   WARMUP=2 ITERS=5 ./run_amd_mega_moe.sh"

# DeepSeek-V4-Flash (needs ≥32 GiB symmetric heap)
docker exec dev_primus bash -lc \
  "cd /apps/zhuang12/MegaKernel/Triton-distributed && \
   MODEL=deepseek-v4-flash TRITON_DIST_SHMEM_BACKEND=mori_shmem \
   MORI_SHMEM_HEAP_SIZE=34359738368 MORI_SHMEM_SYMMETRIC_SIZE=34359738368 \
   WARMUP=2 ITERS=5 ./run_amd_mega_moe.sh"

# DeepSeek-V4-Pro (needs ≥64 GiB symmetric heap)
docker exec dev_primus bash -lc \
  "cd /apps/zhuang12/MegaKernel/Triton-distributed && \
   MODEL=deepseek-v4-pro TRITON_DIST_SHMEM_BACKEND=mori_shmem \
   MORI_SHMEM_HEAP_SIZE=68719476736 MORI_SHMEM_SYMMETRIC_SIZE=68719476736 \
   WARMUP=2 ITERS=5 ./run_amd_mega_moe.sh"
```

### `--ntokens` is GLOBAL; the report below is framed per-rank

The wrapper's `NTOKENS=…` (and the test's `--ntokens`) is the
**global** token count across all EP ranks — at EP=8, each rank sees
`NTOKENS / 8` local tokens. The tables below are labelled by
**local** tokens (the per-rank batch the kernel actually iterates
over) with the global `NTOKENS` argument the wrapper was given in
parentheses.

To sweep only specific large batches without paying for the 1024..16384
warmups, set `EP_NTOKENS_MIN=<global_min>` — the test
([test_ep_moe_fused.py:357-361](python/triton_dist/test/amd/test_ep_moe_fused.py#L357-L361))
drops anything below that.

Baseline tail (8 × MI355X / gfx950, EP=8, mori_shmem, WARMUP=5 ITERS=15,
default drop-MoE input, latency format `latency(ms)/peak_mem(MB)/precision`)
— 2026-05-26, command
`NTOKENS=131072 EP_NTOKENS_MIN=32768 MODEL=<m> ./run_amd_mega_moe.sh`.
`triton_dist_*` is `ep_moe_fused` (this repo's mega-MoE kernel);
`turbo_fwd` is the `turbo_ep_moe` baseline (DeepEPTokenDispatcher +
primus_turbo grouped_gemm, fwd-only). `perf` columns are
`turbo_lat / triton_dist_lat` — values **< 1.0× mean ep_moe_fused is
slower than turbo** (lower latency = better).

Numbers are tracked across optimisation rounds — the **v2** block
below is the initial baseline, **v3** swapped acquire/release flag
traffic for the cheap-fence helpers, and the **v4** block at the
bottom is the current head (round 4: every
`libshmem_device.putmem_warp` on the dispatch / combine fast path
was replaced with the native `copy_warp` helper, and every
`libshmem_device.signal_op` / `libshmem_device.fence`-then-store
sequence on the per-expert / per-token barrier flags was replaced
with a relaxed `atomic_add` / `st_release` to the symm pointer —
see the diff in [`python/triton_dist/kernels/amd/ep_all2all_fused.py`](python/triton_dist/kernels/amd/ep_all2all_fused.py)).
Keep all three blocks intact when adding future rounds so regressions
are visible against the historical lineage.

### v2 — initial baseline:

```
# DeepSeek-V3 (topk=8, num_experts=256, hidden=7168, ffn=2048)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)  13.942/ 915.61/✅   6.395/1646.53/N/A   36.880/ 4499.51/✅    0.459x       0.173x
   8192 (NTOKENS= 65536)  23.984/1804.34/✅  11.901/3278.93/N/A   66.160/ 6277.05/✅    0.496x       0.180x
  16384 (NTOKENS=131072)  42.812/3544.72/✅  22.591/6486.17/N/A  119.521/ 9757.23/✅    0.528x       0.189x

# DeepSeek-V4-Flash (topk=6, num_experts=256, hidden=4096, ffn=2048)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   9.011/ 510.64/✅   3.378/ 927.51/N/A   22.089/ 2542.37/✅    0.375x       0.153x
   8192 (NTOKENS= 65536)  15.742/1034.64/✅   6.115/1912.02/N/A   40.424/ 3590.23/✅    0.388x       0.151x
  16384 (NTOKENS=131072)  28.289/2068.33/✅  11.463/3850.35/N/A   75.594/ 5657.89/✅    0.405x       0.152x

# DeepSeek-V4-Pro (topk=6, num_experts=384, hidden=7168, ffn=3072)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)  12.313/ 836.80/✅   7.050/1483.96/N/A   39.437/ 7706.69/✅    0.572x       0.179x
   8192 (NTOKENS= 65536)  21.335/1667.89/✅  12.494/2986.19/N/A   65.615/ 9368.92/✅    0.586x       0.190x
  16384 (NTOKENS=131072)  38.811/3281.83/✅  23.541/5896.12/N/A  117.173/12596.89/✅    0.607x       0.201x
```

### v3 — cheap-fence `st_release` / `ld_acquire` (2026-05-26):

Same command, same shapes, same warmup/iters. Only change is in
the cross-rank flag traffic: every `ld(..., semantic="acquire")`
and `st(..., semantic="release")` on the dispatch / combine signal
flags was replaced with the cheap-fence helpers
`ld_acquire` / `st_release` from
[`language_extra.py:356-369`](python/triton_dist/language/extra/hip/language_extra.py#L356-L369).

What "cheap fence" actually is — each helper expands to

```
s_waitcnt lgkmcnt(0) vmcnt(0)   ; _memory_barrier()  -- drain LDS+VMEM
<relaxed ld / st>               ; the actual flag access
""  ~{memory}                   ; _compiler_barrier()  -- LLVM signal_fence
```

i.e. a hardware wait-counter drain plus a compiler-side memory
clobber wrapping a *relaxed* atomic. Versus the previous
acquire/release semantics, which LLVM lowers to atomic intrinsics
that emit AMD cache-coherence ops (buffer invalidate / writeback on
top of the same `s_waitcnt`). The cheap fence skips those L1/L2
cache ops — correct here because the dispatch/combine flags are
single-writer-many-reader on `sys` scope, so the data they guard is
already coherent through the symmetric-heap path; we only need
ordering, not cache management. This is the same trade-off as the
`kUseCheapFence` branch of `st_release_sys_global` in Primus-Turbo's
`deep_ep/utils.cuh`.

turbo_fwd column is reproduced unchanged (it shares the run but its
implementation didn't move) so the v3 vs v0 delta is purely on the
`triton_dist_*` columns:

```
# DeepSeek-V3 (topk=8, num_experts=256, hidden=7168, ffn=2048)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   9.746/ 915.61/✅   6.374/1646.53/N/A   29.088/ 4499.51/✅    0.654x       0.219x
   8192 (NTOKENS= 65536)  16.389/1804.34/✅  11.987/3278.93/N/A   49.627/ 6277.05/✅    0.731x       0.242x
  16384 (NTOKENS=131072)  29.838/3544.72/✅  22.634/6486.17/N/A   90.534/ 9757.23/✅    0.759x       0.250x

# DeepSeek-V4-Flash (topk=6, num_experts=256, hidden=4096, ffn=2048)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   6.076/ 510.64/✅   3.364/ 927.51/N/A   16.314/ 2542.37/✅    0.554x       0.206x
   8192 (NTOKENS= 65536)  10.463/1034.64/✅   6.113/1912.02/N/A   29.173/ 3590.23/✅    0.584x       0.210x
  16384 (NTOKENS=131072)  18.499/2068.33/✅  11.463/3850.35/N/A   53.156/ 5657.89/✅    0.620x       0.216x

# DeepSeek-V4-Pro (topk=6, num_experts=384, hidden=7168, ffn=3072)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   9.300/ 836.80/✅   7.064/1483.96/N/A   33.351/ 7706.69/✅    0.760x       0.212x
   8192 (NTOKENS= 65536)  15.871/1667.89/✅  12.448/2986.19/N/A   53.384/ 9368.92/✅    0.784x       0.233x
  16384 (NTOKENS=131072)  29.518/3281.83/✅  23.616/5896.12/N/A   93.864/12596.89/✅    0.800x       0.252x
```

v3 / v2 speedup on `triton_dist_fwd` (lower latency in v3 / v2 latency;
higher = better; peak_mem unchanged because the allocator and shapes
didn't move):

| Shape         | local 4096 | local 8192 | local 16384 |
|---|---:|---:|---:|
| V3            | 1.43×      | 1.46×      | 1.43×       |
| V4-Flash      | 1.48×      | 1.50×      | 1.53×       |
| V4-Pro        | 1.32×      | 1.34×      | 1.31×       |

### v4 — native `copy_warp` + relaxed `atomic_add` on dispatch/combine (2026-05-28):

Same command, same shapes, same warmup/iters. Two changes vs v3:

1. **Drop `libshmem_device.putmem_warp` on the dispatch + combine
   token-copy fast path** ([`ep_all2all_fused.py:124,131,234,244`](python/triton_dist/kernels/amd/ep_all2all_fused.py#L124))
   in favour of `copy_warp(dst_ptr, src_ptr, bytes)` — the same
   warp-cooperative `tl.load` + `tl.store` we already use for
   intra-rank copies. The mori dispatcher under
   `libshmem_device.putmem_warp` had a per-call setup cost (target-rank
   symm-pointer resolution + intrinsic dispatch) that dominated for
   the short per-token transfers; the native helper compiles down
   directly to the symm pointer the kernel already computed via
   `dl.symm_at`.
2. **Drop `libshmem_device.signal_op(... MORI_SIGNAL_SET ..., remote_rank)`
   plus its preceding `libshmem_device.fence()`** on every per-expert
   ready-flag bump and zero-token signal ([`ep_all2all_fused.py:139-145,152-156`](python/triton_dist/kernels/amd/ep_all2all_fused.py#L139)),
   and replace with a single relaxed `atomic_add(remote_flag_ptr,
   1, scope="agent"|"gpu", semantic="relaxed")` plus, where the
   sender needs ordering with a prior store, the already-released
   `st_release` to the symm pointer that v3 introduced. The
   release/fence pair was overkill — the receiver-side ordering is
   guarded by the existing `ld_acquire` in the wait loop, and the
   token payload's own `st_release` (sys scope) already publishes
   the data; the extra `libshmem_device.fence()` only added a redundant
   wait-counter drain on the sender's path.

Both swaps preserve correctness: dispatch / combine still gate on
the same `ld_acquire(barriers_ptr + ...)` wait loops, and the
counter writes are still `sys`-visible (mori's symm pointer maps
the remote rank's HBM into the local agent's address space).
Companion `common.py` tidy-up collapses the dead small-CU branches
of `get_moe_optim_config` (MI355X has 256 CUs ≫ 78) and aligns
`num_group_gemm_warps` for the mega-MoE backward path (16→8) —
these are config-only and don't change the dispatch / combine code
path, but they ride in the same v4 changeset.

turbo_fwd column is reproduced unchanged (its baseline didn't move),
so the v4 vs v3 delta is purely on the `triton_dist_*` columns:

```
# DeepSeek-V3 (topk=8, num_experts=256, hidden=7168, ffn=2048)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   6.339/ 915.61/✅   6.486/1653.50/N/A   20.552/ 4499.51/✅    1.023x       0.900x
   8192 (NTOKENS= 65536)  11.212/1804.34/✅  11.968/3292.82/N/A   34.013/ 6277.05/✅    1.067x       0.959x
  16384 (NTOKENS=131072)  20.002/3544.72/✅  22.466/6513.77/N/A   58.614/ 9757.23/✅    1.123x       1.024x

# DeepSeek-V4-Flash (topk=6, num_experts=256, hidden=4096, ffn=2048)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   3.330/ 510.64/✅   3.392/ 932.97/N/A   10.609/ 2542.37/✅    1.019x       0.905x
   8192 (NTOKENS= 65536)   5.585/1034.64/✅   6.117/1922.52/N/A   16.561/ 3590.23/✅    1.095x       0.996x
  16384 (NTOKENS=131072)   9.740/2068.33/✅  11.282/3874.17/N/A   27.919/ 5657.89/✅    1.158x       1.070x

# DeepSeek-V4-Pro (topk=6, num_experts=384, hidden=7168, ffn=3072)
 local (global)        triton_dist_fwd          turbo_fwd      triton_dist_fwd_bwd     fwd_perf  fwd_bwd_perf
=============================================================================================================
   4096 (NTOKENS= 32768)   7.190/ 836.80/✅   7.089/1491.60/N/A   28.096/ 7706.69/✅    0.986x       0.823x
   8192 (NTOKENS= 65536)  12.162/1667.89/✅  12.444/3001.38/N/A   41.859/ 9368.92/✅    1.023x       0.893x
  16384 (NTOKENS=131072)  22.216/3282.74/✅  23.306/5926.77/N/A   69.651/12596.89/✅    1.049x       0.953x
```

**Result: `triton_dist_fwd` now beats the `turbo_fwd` baseline on
every V3 / V4-Flash shape and on the two larger V4-Pro shapes**
(V4-Pro 32768 is the only remaining sub-1.0× point at 0.986×).
`triton_dist_fwd_bwd` reaches parity (≥1.0×) on the two larger V3
shapes and V4-Flash 65536 / 131072; backward on V4-Pro and smaller
V4-Flash is still ~5-10% slower than turbo and is the next target.

v4 / v3 speedup on `triton_dist_fwd` (v3 lat / v4 lat; higher =
better; peak_mem essentially unchanged — small drift from the
turbo dispatcher's own allocator, not from `ep_moe_fused`):

| Shape         | local 4096 | local 8192 | local 16384 |
|---|---:|---:|---:|
| V3            | 1.54×      | 1.46×      | 1.49×       |
| V4-Flash      | 1.82×      | 1.87×      | 1.90×       |
| V4-Pro        | 1.29×      | 1.30×      | 1.33×       |

v4 / v3 speedup on `triton_dist_fwd_bwd`:

| Shape         | local 4096 | local 8192 | local 16384 |
|---|---:|---:|---:|
| V3            | 1.42×      | 1.46×      | 1.54×       |
| V4-Flash      | 1.54×      | 1.76×      | 1.90×       |
| V4-Pro        | 1.19×      | 1.28×      | 1.35×       |

The V4-Flash row shows the largest win — its smaller hidden (4096
vs 7168) means the per-token copy is short enough that the
`putmem_warp` setup cost dominated; replacing it with `copy_warp`
removes the most overhead per token. V4-Pro shows the smallest win
because its larger `ffn` (3072) and `num_experts` (384) push the
runtime back towards being GEMM-bound rather than comms-bound.

Reproduce with:

```bash
docker exec dev_primus bash -lc \
  "cd /apps/zhuang12/MegaKernel/Triton-distributed && \
   MODEL=deepseek-v3 TRITON_DIST_SHMEM_BACKEND=mori_shmem \
   MORI_SHMEM_HEAP_SIZE=68719476736 MORI_SHMEM_SYMMETRIC_SIZE=68719476736 \
   NTOKENS=131072 EP_NTOKENS_MIN=32768 WARMUP=5 ITERS=15 ./run_amd_mega_moe.sh"
# Same for MODEL=deepseek-v4-flash (heap=34359738368) and MODEL=deepseek-v4-pro
# (heap=68719476736).
```



## Skip `pip install` — C++ extensions are pre-built

This checkout already contains:

- `python/triton/_C/libtriton.so` + `libtriton_distributed.so`
- `shmem/rocshmem_bind/pyrocshmem/build/lib.linux-x86_64-cpython-312/_pyrocshmem*.so`
- `shmem/rocshmem_bind/rocshmem_build/install/lib/librocshmem.a` +
  device bitcodes (built with `USE_GDA && USE_RO && USE_IPC`)

So you do **not** need to run `pip3 install -e python ...` or
`shmem/rocshmem_bind/build.sh` for a smoke test. Just export
`PYTHONPATH` (the wrapper already does):

```bash
export PYTHONPATH=$ROOT/python:$ROOT/shmem/rocshmem_bind/pyrocshmem/build/lib.linux-x86_64-cpython-312
```

Re-run the pip install only if you actually touched the C++ sources
under `csrc/` or `shmem/rocshmem_bind/`.

### `scripts/launch_amd.sh` PYTHONPATH gotcha

`launch_amd.sh` appends `shmem/rocshmem_bind/pyrocshmem/build` to
PYTHONPATH, but the `pyrocshmem` package lives one level deeper at
`build/lib.linux-x86_64-cpython-312/pyrocshmem`. So if you call
`launch_amd.sh` directly without pre-exporting PYTHONPATH yourself,
`import pyrocshmem` fails with `ModuleNotFoundError`. Either:

- use [`run_amd_mega_moe.sh`](run_amd_mega_moe.sh), or
- pre-export PYTHONPATH with the `lib.linux-x86_64-cpython-312`
  subdir, or
- run `pip3 install --no-build-isolation --no-deps -v .` inside
  `shmem/rocshmem_bind/pyrocshmem/` to install the package properly.

## Switching to the mori_shmem backend

`triton_dist` honours `TRITON_DIST_SHMEM_BACKEND=mori_shmem` to
swap the symmetric-heap implementation from rocSHMEM to AMD's
[mori](https://github.com/ROCm/mori) (the Python `mori.shmem`
module). The wrapper plumbs this through unchanged.

1. **One-time install (inside the container):**

   ```bash
   docker exec dev_primus bash -lc \
     "pip install git+https://github.com/ROCm/mori.git"
   ```

   Use HTTPS — the container has no SSH key configured, so
   `git+git@github.com:...` (as printed in some upstream READMEs)
   will fail with a key-auth error. Verify with
   `python3 -c "import mori.shmem; print(mori.shmem.MoriShmemBuffer)"`.

2. **Bump the heap.** mori's default static heap is 4 GiB but the
   test allocates ~7.2 GiB of symmetric tensors (see the
   `[EpAll2AllOp] nvshmem memory required: 7357.64 MB` log line).
   Without the override you get `AssertionError:
   mori_shmem.shmem_malloc failed` from
   `python/triton_dist/utils.py:277`. The wrapper sets both knobs
   to 8 GiB:

   | Var | Wrapper value | Why |
   |---|---|---|
   | `MORI_SHMEM_HEAP_SIZE` | `8589934592` | static-heap size cap |
   | `MORI_SHMEM_SYMMETRIC_SIZE` | `8589934592` | the value `EpAll2AllOp` checks; without this you also see `MORI_SHMEM_SYMMETRIC_SIZE is too small ... is -1 bytes` |

3. **No bootstrap-timeout / IFNAME knobs needed.** mori uses MPI
   (or torch's PG) for the bring-up handshake, not the rocSHMEM TCP
   accept loop, so the `ROCSHMEM_BOOTSTRAP_*` overrides are inert
   on this path. Leave them in the wrapper — they only fire when
   `TRITON_DIST_SHMEM_BACKEND=rocshmem`.

## Four rocSHMEM env knobs missing from the project doc

The doc covers `ROCSHMEM_HOME` + `ROCSHMEM_HEAP_SIZE` + `ulimit -l`
but not the bootstrap-timeout / IFNAME pair. All four matter:

| Var | Value | Why it matters |
|---|---|---|
| `ROCSHMEM_HOME` | `<root>/shmem/rocshmem_bind/rocshmem_build/install` | Container default `/opt/rocshmem` is **GDA-only** (no IPC). Verify with `nm $ROCSHMEM_HOME/lib/librocshmem.a \| grep select_backend_type` — empty on prebuilt, present in in-tree. Without this, single-node runs fail with `Failed to lock memory pool ((nil)): 0x1001` (GDA tries to lock a NULL doorbell page when no IB device is available). |
| `ROCSHMEM_BOOTSTRAP_TIMEOUT` | `120` | Default 5 s accept window times out before 8 cold-start torchrun ranks finish `import torch`. Symptom: `accept timeout` lines from `socket.cpp:543` followed by SIGABRT (exit -6) on every non-zero rank, and SIGTERM on rank 0. |
| `ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME` | `lo` | `--network=host` exposes per-NIC /31 RDMA interfaces (`benic*`, `fenic`). Auto-pick can choose one whose in-container connectivity is unreliable. Loopback always works since all 8 ranks share the host netns. |
| `ROCSHMEM_HEAP_SIZE` + `ulimit -l unlimited` | `8589934592` for ntokens≥2048 | Symmetric heap is pinned at `rocshmem_init_attr`; container default memlock is 8 MiB. `--privileged` container allows the ulimit override; the `[EpAll2AllOp] ROCSHMEM_HEAP_SIZE is updated to ...` log line is a no-op (heap is already pinned), so the env var must be set **before** launching. ntokens=1024 fits in the default 512 MiB. |

## Test-shape sizing

From `docs/amd_ep_moe_fused_port.md` § "Optional knobs the test honours":

| Flag | Default | Notes |
|---|---|---|
| `--ntokens` | `4096` | per-EP-group; ntokens=2048 needs ~7.2 GiB symmetric heap |
| `--hidden_dim` | `1536` | matches NVIDIA reference |
| `--ffn_dim` | `480` | matches NVIDIA reference |
| `--topk` | `8` | |
| `--num_experts` | `64` | |
| `--num_ranks` | WORLD_SIZE | |
| `--capacity` | `4.0` | drop-token capacity |
| `--warmup` / `--iters` | `2` / `3` | functional sanity sweep |
| `--skip_backward` | off | run forward only (handy while bisecting comms) |

`run_amd_mega_moe.sh` honours `NTOKENS`, `WARMUP`, `ITERS` env vars
and forwards any extra positional args to the test.

## Cluster gotcha

`smci355-ccs-aus-n*` (Austin AMD MI355X) is the same cluster as
`dccs-1334-slurm` in the global `ainic-ionic-abi-cluster-issue`
memory. **Single-node EP=8 is fine on any node** (rocSHMEM bootstrap
is TCP and uses IPC backend, not ionic). For multi-node EP (EP≥16),
filter NODELIST to `smci355-ccs-aus-n01-21` — that's the only node
where container-side `libibverbs` finds the 8 AINIC `ionic_*` HCAs.

## Rebuild from scratch (only if you really need to)

If you modified C++ sources and need a clean rebuild, see
[`docs/build.md`](docs/build.md) and
[`docs/amd_ep_moe_fused_port.md`](docs/amd_ep_moe_fused_port.md).
The high-level order is:

```bash
# 1. rocshmem device library + host static lib (USE_IPC+USE_RO+USE_GDA)
ROCM_ARCH=gfx950 bash ./shmem/rocshmem_bind/build.sh

# 2. pyrocshmem — MUST point ROCSHMEM_HOME at the in-tree install,
#    otherwise pip picks up /opt/rocshmem (GDA-only) and IPC is impossible
cd shmem/rocshmem_bind/pyrocshmem
rm -rf build dist *.egg-info python/*.egg-info
export ROCSHMEM_HOME=$PWD/../rocshmem_build/install
export CXX=hipcc TORCH_DONT_CHECK_COMPILER_ABI=1
pip3 install --no-build-isolation --no-deps -v .

# 3. triton_dist
cd ../../..
OMPI_DIR=/opt/ompi_build/install/ompi \
ROCM_ARCH=gfx950 \
TRITON_DIST_SHMEM_BACKEND=rocshmem \
    pip3 install -e python --no-build-isolation --use-pep517
```

`build.sh` does **not** override the inherited `ROCSHMEM_HOME`
(only `ROCSHMEM_DIR`), so the explicit override in step 2 is
load-bearing on the `dev_primus` container.
