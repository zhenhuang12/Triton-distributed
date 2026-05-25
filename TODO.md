# Triton-distributed AMD MoE optimization TODO

Backlog of grouped-GEMM perf items in
`python/triton_dist/kernels/amd/ep_all2all_fused.py`, derived from a
review against the ROCm gfx950 Gluon tutorials
(<https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/a16w16>).
Tutorial reference arc: a16w16 v0→v9 takes a 520 TFLOPS naive kernel to
~1489 TFLOPS (~3×) with the steps catalogued below.

Production shapes used to size each item:
`test_ep_moe_fused.py` — hidden=1536, ffn=480, topk=8, E=64, EP=8,
ntokens up to 8192 → avg per-expert M ≈ 1024.

GEMM1 (post-dispatch): `M=1024, K=1536, N=480`
GEMM2 (combine):       `M=1024, K=480,  N=1536`

Current tile config (`function/amd/ep_moe_fused.py:273-278` and
`function/amd/common.py:449-508`):
`BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=256, GROUP_SIZE_M=4`,
`num_stages=3`, `num_group_gemm_warps ∈ {4, 16}` depending on preset.
No `matrix_instr_nonkdim`, no `kpack`, no `waves_per_eu`, no XCD remap,
no `cache_modifier` set.

---

## Cheap wins (do first)

### GG-1. Gate K-mask on `K_EVEN: tl.constexpr` in `dot_k_const`
- **File**: `python/triton_dist/kernels/amd/ep_all2all_fused.py:559-593`
- Both branches of `dot_k_const` always emit the K-direction mask
  `(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)`. For the
  production shape (K=1536, BLOCK_SIZE_K=256, 6 even tiles) the mask is
  dead weight and forces a per-iter compare in the MFMA pipeline.
- **Fix**: add `K_EVEN: tl.constexpr` arg; when true, drop the K-mask on
  both A and B loads. Pass `K_EVEN = (K % BLOCK_SIZE_K == 0)` from the
  outer kernel.
- **Why**: mirrors the v0→v1 (`buffer_load`) tutorial step — collapses
  branch count in the hot loop, lets the AMD backend emit pure
  `buffer_load` with no per-iter compare.
- **Expected**: a few %.

### GG-2. XCD-aware PID remap (MI350/MI355 has 8 XCDs)
- **File**: `python/triton_dist/kernels/amd/ep_all2all_fused.py:597-649`
- `tl.swizzle2d(local_pid_m, pid_n, ...)` only orders tiles *within* an
  expert. Across experts the schedule is plain round-robin across all
  256 CUs, scattering adjacent groups across the 8 XCDs (each with its
  own private L2).
- **Fix**: insert an XCD remap at the top of
  `tile_kernel_moe_grouped_gemm_nk_const`, before the `pid_m, pid_n`
  split. Pattern from a16w16 v9: cluster `pid` so consecutive `pid`s
  land on the same XCD.
- **Why**: tutorial v9 reduces L2 misses ~5M → ~4M → higher sustained
  clocks → final uplift to 1489 TFLOPS on top of the saturated hot
  loop.
- **Expected**: 2–5%. Free win.

### GG-3. Per-GEMM tile config (GEMM2 wants wider N)
- **File**: `python/triton_dist/function/amd/ep_moe_fused.py:273-278,
  341-346`
- GEMM1 N=480 — current 64×128 is fine (3.75 tiles, edge waste ≈ 6%).
- GEMM2 N=1536 — 128 leaves 12 N-tiles per M-row; widening to 256 cuts
  that to 6 and amortizes the per-tile barrier/scheduler overhead.
- **Fix**: specialize the combine-path launch to
  `BLOCK_SIZE_M=64, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128` and measure.
  The kernel already takes `BLOCK_SIZE_*` as `tl.constexpr` — only the
  launch site needs to change.
- **Why**: matches the tutorial's preference for 256-wide tiles on
  large-N shapes (a16w16 uses 256×256×64).
- **Expected**: medium — combine GEMM is the bigger of the two
  (1024 × 480 × 1536 = ~755 MFLOPs vs GEMM1's ~755 MFLOPs, but combine
  also feeds the reduce-scatter so latency is more visible).

### GG-4. Sweep `matrix_instr_nonkdim` and `kpack`
- **File**: `python/triton_dist/function/amd/ep_moe_fused.py` (launch
  sites).
- Triton's AMD backend exposes `matrix_instr_nonkdim ∈ {16, 32}` (MFMA
  tile shape) and `kpack ∈ {1, 2}` (k-direction packing into LDS) as
  kernel-launch kwargs. Current launches set neither → defaults pick a
  generic config that may not match MI355X.
- **Fix**: 2 × 2 sweep on each of the GEMM launches; pick the winner
  per shape and pin it.
- **Why**: these are the tutorial's hand-tuned v3 (LDS layout) and v7
  (sliceN, MFMA tile selection) knobs surfaced at the Triton DSL level.
- **Expected**: 5–15% if the default is wrong.

---

## Larger structural items

### GG-5. Inspect Triton-AMD lowering for `buffer_load … lds`
- **File**: emitted assembly for
  `tile_kernel_moe_grouped_gemm_nk_const`.
- a16w16 v2 (HBM→LDS direct async copy, skipping registers) is worth
  ~150 TFLOPS and ~100 VGPRs in the tutorial. In Triton this is a
  backend choice, not a user knob — verify with
  `TRITON_DEBUG_DUMP=1` / `--print-ttgir` / `rocprof` whether the
  emitted code uses `buffer_load_b128 ... lds` or staged through VGPR.
- **If not**: file a Triton issue / consider a manual `tl.inline_asm`
  shim. **If yes**: nothing to do, just confirm.
- **Expected**: large if not already on, zero if it is.

### GG-6. Larger N tile (256) on saturated experts for v7-style MFMA eff
- **File**: `python/triton_dist/kernels/amd/ep_all2all_fused.py:597-716`
- Tutorial v7 (sliceN at N=128 halves of a 256-wide tile) reaches **98%
  MFMA efficiency** with `amdgcnas` peephole; v8 (sliceMN) holds eff
  while removing HBM-contention stalls. Current 64×128 sits below this
  ceiling — most likely 70–85% efficient.
- **Fix path**: route saturated experts (split_size ≥ 128) through a
  64×256×128 tile; keep 64×128×256 as the fallback for partially-full
  experts. Requires a small dispatcher in the persistent loop.
- **Caveat**: 64×256 fp32 accumulator = 64 KB → ~256 VGPR/lane. Need to
  verify no spill on MI355X (512 VGPR/lane budget) before committing.
- **Expected**: 10–20% on the GEMM portion when most experts are
  saturated.

### GG-7. Custom LLIR scheduler / `amdgcnas` post-pass
- The tutorial gets from ~76% (v5, default scheduler) → 98% (v7 +
  custom LLIR scheduler + `amdgcnas` AGPR↔VGPR peephole) MFMA
  efficiency. Triton-AMD currently has no equivalent hook.
- **Action**: track upstream Triton AMD scheduler work; revisit when a
  v5-style LLIR pass lands. Until then, ~85% MFMA eff is the practical
  ceiling for this kernel.
- **Expected**: up to ~20% headroom — gated on upstream.

### GG-8. Small-expert fast path (BLOCK_M=32)
- **File**: `python/triton_dist/kernels/amd/ep_all2all_fused.py:693-698`
- When `row_remain < BLOCK_SIZE_M`, the masked branch runs the full
  64-row MFMA but writes ≤63 valid rows. For workloads with heavy load
  imbalance across experts (skewed routing) this wastes MFMA lanes.
- **Fix**: add a 32-row fast path; pick at scheduler time based on
  `split_size` per expert.
- **Expected**: only matters under skew; defer until profiled.

---

## Already well-aligned — do not regress

- `num_stages=3` covers tutorial v4 (double-buffered global prefetch,
  the **+47%** single biggest win) and v5 (3-stage local prefetch) at
  the Triton DSL level. Realised quality bounded by Triton-AMD's
  scheduler, not by tutorial-grade hand LLIR.
- `tl.swizzle2d(local_pid_m, pid_n, …, GROUP_SIZE_M=4)` is the right
  within-expert L2-friendly traversal — matches the tutorial's
  `GROUP_SIZE_M` tuning (optimum minimizes `GM + ⌈P/GM⌉`; GM=4 is
  optimal for P=32 program count).
- Persistent mega-kernel structure (dispatch ↔ GEMM ↔ combine overlap
  via per-token release barriers, lines 1018-1283) is **beyond** the
  tutorial's scope and is correct for MoE — no comparable tutorial
  pattern to borrow from.

## Caveats on the tutorial baseline

- The gfx950-gluon-tutorials repo currently has **no MoE / grouped-GEMM
  example** (README lists it as planned). All recommendations transfer
  the dense-GEMM a16w16 lessons; a real apples-to-apples MoE port would
  need `rocprof --stats` confirmation per item.
- The tutorial is written in **Gluon**, not stock Triton. Triton-AMD
  exposes a subset of Gluon's knobs (e.g., no user LLIR scheduler, no
  `amdgcnas` post-pass). Items GG-1..4 are all Triton-reachable; GG-7
  is not.
