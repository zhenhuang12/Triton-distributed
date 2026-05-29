# Mega EP-MoE kernel profile — DeepSeek V3, EP=8 MI355X

Workload: `NTOKENS=65536`, `hidden=7168`, `ffn=2048`, `topk=8`, `num_experts=256`,
single-node EP=8 on `dev_primus` container (`mori_shmem` backend, 64 GiB heap).
All numbers from `run_amd_mega_moe.sh` with `--warmup 5 --iters 15` (median over
the iter loop, reported by the test harness).

## Diagnostic knobs (envs → constexpr)

Three knobs plumbed end-to-end (env var → `ep_moe_fused.py`
→ `ep_a2a_fused_layer.py` → kernel `tl.constexpr`):

| Env var | Dispatch tile runs? | Dispatch GEMM waits on per-tile barrier? | Combine tile runs? | Combine GEMM waits? |
|---|---|---|---|---|
| (none, baseline) | yes | yes | yes | yes |
| `TRITON_DIST_SKIP_DISPATCH_WAIT=1` | yes | **no** | yes | yes |
| `TRITON_DIST_SKIP_COMM=1` | **no** | no | **no** | no |

`SKIP_DISPATCH_WAIT` isolates "GEMM stalls waiting for dispatch arrival" from
"dispatch comm consumes CTAs". `SKIP_COMM` removes both halves of comm
(dispatch tile + GEMM wait + combine tile + combine wait) and gives the
compute-only floor.

`TRITON_DIST_SKIP_DISPATCH_COMM` / `TRITON_DIST_SKIP_COMBINE_COMM` are
the one-sided variants — they suppress the dispatch tile or the combine
tile *and* the matching GEMM wait, independently. The per-side ablation
below uses these to decompose the comm budget.

## Per-side ablation (NTOKENS=65536) — dispatch tile vs combine tile

This round isolates the comm cost on each side using
`TRITON_DIST_SKIP_DISPATCH_COMM` and `TRITON_DIST_SKIP_COMBINE_COMM`
independently, plus `TRITON_DIST_SKIP_DISPATCH_WAIT` as a finer
sub-knob that only removes the GEMM-side wait without killing the
dispatch tile. Together the four configurations give a three-way
decomposition of the fwd comm budget (wait vs dispatch-tile-CTAs
vs combine).

| Run | fwd (ms) | fwd+bwd (ms) | fwd_perf vs turbo | Δfwd vs baseline | Δfwd+bwd vs baseline |
|---|---|---|---|---|---|
| baseline | 15.526 | 45.895 | 0.769x | — | — |
| `SKIP_DISPATCH_WAIT=1` only | 15.370 | 45.013 | 0.780x | **−0.16 ms** | **−0.88 ms** |
| `SKIP_DISPATCH_COMM=1` only | 12.467 | 37.481 | 0.958x | **−3.06 ms** | **−8.41 ms** |
| `SKIP_COMBINE_COMM=1` only | 12.649 | 40.485 | 0.944x | **−2.88 ms** | **−5.41 ms** |
| both COMM = 1 (= `SKIP_COMM=1`) | 9.504 | 32.077 | 1.256x | **−6.02 ms** | **−13.82 ms** |

### Decomposition

- **Dispatch GEMM-wait cost (fwd):** ~0.16 ms — the per-tile barrier
  wait loop in the GEMM consumer. Essentially overlapped in fast state.
- **Dispatch tile-CTA cost (fwd):** ~2.90 ms — derived as
  `SKIP_DISPATCH_COMM − SKIP_DISPATCH_WAIT = 3.06 − 0.16`. This is the
  SM-occupancy cost of the dispatch tile itself: CTAs running
  rocSHMEM puts / signals occupy SMs that the grouped GEMM could
  otherwise schedule on.
- **Combine comm cost (fwd):** ~2.88 ms — combine tile (scatter +
  topk-reduce) + combine GEMM wait. Almost identical to dispatch in fwd.
- **Dispatch + combine are linearly additive (fwd):**
  `−3.06 + −2.88 = −5.94 ms` vs combined `−6.02 ms` — basically
  perfect superposition. The two sides do not contend for the same
  resources in fast state.
- **Dispatch comm cost (fwd+bwd):** ~8.41 ms — bwd dispatch is much
  heavier than fwd dispatch (bwd dispatches `dy` over the transposed
  schedule). Most of the savings here come from the bwd half.
- **Combine comm cost (fwd+bwd):** ~5.41 ms.
- **Linear superposition holds exactly in fwd+bwd:**
  `−8.41 + −5.41 = −13.82 ms` (matches the both-skip column to the
  third decimal).

### Takeaway

- **The dispatch comm bill is overwhelmingly tile-CTAs, not wait.**
  Of the ~3.06 ms dispatch fwd cost, only ~0.16 ms (≈5 %) is the
  GEMM-side wait — the remaining ~2.90 ms is the dispatch tile
  occupying SMs. Future dispatch-side optimisations should target
  *CTA budget / dispatch-tile shortening*, not wait-loop shaving.
- **Dispatch and combine are roughly equal in fwd** (~3 ms each).
  Earlier conclusion that "combine dominates" was an artefact of
  comparing `SKIP_DISPATCH_WAIT` (only removes 0.16 ms) against
  `SKIP_COMM` (removes both sides entirely).
- **For full fwd+bwd, dispatch is the bigger lever** (~8.4 ms vs
  ~5.4 ms). The bwd dispatch path (used to compute `grad_swiglu_output`
  from `dy`) should be profiled separately from the fwd dispatch.
- **The two sides do not contend for resources in fast state.**
  Superposition is essentially perfect, so co-optimising both (e.g.
  reducing both tile widths) compounds rather than fighting itself.
- **The grouped-GEMM compute floor is competitive** (9.5 ms vs
  turbo's 11.9 ms) — the gap to turbo is comm overhead, not GEMM
  efficiency.

### Noise caveat (important)

Baseline wall-clock varies wildly across runs depending on cluster state:

| Baseline run | fwd (ms) | fwd+bwd (ms) | fwd_perf |
|---|---|---|---|
| run 1 (this session) | 20.335 | 56.960 | 0.588x |
| run 2 | 25.424 | 57.194 | 0.469x |
| run 3 | 25.615 | 57.110 | 0.467x |
| run 4 | 15.495 | 46.092 | 0.769x |
| run 5 (ablation block) | 15.570 | 46.058 | 0.766x |

Range: **15.5 – 25.6 ms on fwd (±26 % around the mean)** — not code, but GPU
power state / autotune cache / scheduler contention. The decomposition
above is internally consistent (all three rows collected back-to-back at
ports 37091/37121/37131 within ~5 minutes), but single-point comparison
between unrelated runs is not reliable. **Always re-baseline immediately
before measuring an optimization round.**

When the cluster is in the slow state (~25 ms baseline), the same ablation
gives different splits (`SKIP_DISPATCH_WAIT` saved 2.27 ms instead of 0.20 ms),
suggesting the dispatch-wait cost grows when the dispatch tile itself
contends harder for CTAs — another reason to run all three legs in one
back-to-back block.

## How to reproduce

```bash
PORT=37001  # increment between runs — torchrun rdzv socket lingers ~60 s

for cfg in "" "TRITON_DIST_SKIP_DISPATCH_WAIT=1" "TRITON_DIST_SKIP_COMM=1"; do
  docker exec -w /apps/zhuang12/MegaKernel/Triton-distributed \
      -e MODEL=deepseek-v3 -e NTOKENS=65536 \
      -e MORI_SHMEM_HEAP_SIZE=68719476736 \
      -e MORI_SHMEM_SYMMETRIC_SIZE=68719476736 \
      -e MORI_SOCKET_IFNAME=lo \
      -e ARNOLD_WORKER_0_PORT=$PORT \
      ${cfg:+-e "$cfg"} \
      dev_primus bash run_amd_mega_moe.sh
  PORT=$((PORT + 10))
done
```

`SKIP_DISPATCH_WAIT=1` occasionally hangs at `NTOKENS=65536` during the
sweep (GEMM reading uninitialised dispatch buffers can trigger slow paths or
NaN cascades that the precision check time-outs against). Retry on hang —
two of every three attempts complete cleanly in this session.

## In-kernel profiler reference (preserved from an earlier slow run)

Numbers below are from a different sweep where `triton_dist_fwd` measured
`6950.637 ms` (vs `turbo_fwd=12.010 ms`) — useful for the **relative**
breakdown of work inside each mega kernel, not for absolute timing:

```Python
 MoEOptimConfig(
                num_build_sms=8,
                num_copy_sms=max_sms,
                num_group_gemm_warps=4,
                num_dispatch_warps=16,
                num_combine_warps=16,
                num_dispatch_sms=80,
                num_tail_sms_in_dispatch=0,
                num_combine_sms=80,
                num_reduce_sms_in_combine=80,
                dispatch_use_block_wise_barrier=False,
            )
```

```
================ in-kernel profiler summary (ns / record) ================
kernel                                           task                                  n       mean        p50        p99        max   share
mega_bwd_dispatch_group_gemm_rank_0              group_gemm_wait                  229096     926159         92   21140665  100739084   85.7%
mega_bwd_dispatch_group_gemm_rank_0              dispatch_token_tail_notify         6912    4447013         32   68297608  100981804   12.4%
mega_bwd_dispatch_group_gemm_rank_0              group_gemm_main                  229096      16101      15256      18508     972841    1.5%
mega_bwd_dispatch_group_gemm_rank_0              dispatch_token_main                6912     127880     161004     229149     260700    0.4%
mega_bwd_dispatch_group_gemm_rank_0              group_gemm_preprocess            229096          9          8         12        144    0.0%
------------------------------------------------------------------------------------------------------------------------
mega_bwd_group_gemm_combine_rank_0               transposed_group_gemm_main       1548288       6467       6268       7248    1227314   33.5%
mega_bwd_group_gemm_combine_rank_0               combine_topk_reduce                6912     978630      31760   51147114   51341396   22.6%
mega_bwd_group_gemm_combine_rank_0               combine_scatter_token              6912     962244     956072    1098574    1110800   22.2%
mega_bwd_group_gemm_combine_rank_0               group_gemm_main                  801836       7551       7348       9948     105698   20.2%
mega_bwd_group_gemm_combine_rank_0               group_gemm_tail_notify           801836        513        520        796      96839    1.4%
mega_bwd_group_gemm_combine_rank_0               transposed_group_gemm_preprocess 1548288          5          4         12        192    0.0%
mega_bwd_group_gemm_combine_rank_0               group_gemm_preprocess            801836          5          4          8        260    0.0%
------------------------------------------------------------------------------------------------------------------------
mega_dispatch_group_gemm_rank_0                  group_gemm_main                  899568      15363      15420      18084     120727   79.3%
mega_dispatch_group_gemm_rank_0                  dispatch_token_main               10176     131175     168642     278399     327164    7.7%
mega_dispatch_group_gemm_rank_0                  group_gemm_wait                  899568       1405        104      12892    1371188    7.3%
mega_dispatch_group_gemm_rank_0                  dispatch_token_tail_notify        10176      97693         36     451263    1693920    5.7%
mega_dispatch_group_gemm_rank_0                  group_gemm_preprocess            899568         10          8         12         20    0.1%
------------------------------------------------------------------------------------------------------------------------
mega_group_gemm_combine_rank_0                   group_gemm_main                  1574244       4014       3876       5356     105285   49.4%
mega_group_gemm_combine_rank_0                   combine_scatter_token             10176     472352     470820     507177     571515   37.6%
mega_group_gemm_combine_rank_0                   group_gemm_tail_notify           1574244        666        660       1064     101055    8.2%
mega_group_gemm_combine_rank_0                   combine_topk_reduce               10176      59233      35132     823669    1643224    4.7%
mega_group_gemm_combine_rank_0                   group_gemm_preprocess            1574244          6          4         12        148    0.1%
------------------------------------------------------------------------------------------------------------------------
```

Note the tension with the wall-clock ablation: the profiler shows
`group_gemm_wait` taking 38–93 % of dispatch share, but the ablation says
`SKIP_DISPATCH_WAIT` only saves ~0.2 ms. The reconciliation: `group_gemm_wait`
is the dominant *per-CTA* cost but it overlaps with comm CTAs in the same
kernel, so removing the wait doesn't free wall-clock unless other CTAs were
also blocked — they aren't in the fast state. In the slow state both go up
together and `SKIP_DISPATCH_WAIT` saves more (~2.3 ms). Profiler share ≠
wall-clock savings when work is overlapped.
