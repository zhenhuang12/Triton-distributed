# Performance of Triton-distributed on AMD GPUs

## AllGather GEMM Single Node MI308X
![AG-GEMM](../asset/amd-ag-gemm-intranode-perf.png)

## GEMM ReduceScatter Single Node MI308X
![GEMM-RS](../asset/amd-gemm-rs-intranode-perf.png)

## Warp-level put bandwidth — rocshmem vs mori_shmem (MI355X, intra-node)

Micro-benchmark of the two device calls used by `tile_kernel_dispatch_token_intra_node`
in [ep_all2all_fused.py](../python/triton_dist/kernels/amd/ep_all2all_fused.py):

* `libshmem_device.putmem_warp` — blocking (source-buffer reusable on return)
* `libshmem_device.putmem_nbi_warp` — non-blocking; kernel ends with
  `libshmem_device.fence()` so the measurement includes delivery-to-wire,
  matching the blocking-put semantics.

Driver: [test_putmem_warp_bw.py](../python/triton_dist/test/amd/test_putmem_warp_bw.py).

Setup: single MI355X node (gfx950), 8 ranks, `num_sms=8`, `num_warps=4`
(= 32 put-issuing warps), 10 timed iters / 3 warmup, average over all
56 (src, dst) pairs.

| chunk_bytes | total MiB | rocshmem warp | rocshmem nbi_warp | mori_shmem warp | mori_shmem nbi_warp |
| ----------: | --------: | ------------: | ----------------: | --------------: | ------------------: |
|         256 |       1.0 |          1.71 |       2.34 (+37%) |            2.88 |                2.87 |
|        1024 |       4.0 |          5.42 |       7.79 (+44%) |           27.68 |               27.27 |
|        3072 |      12.0 |          8.45 |      11.45 (+36%) |           27.30 |               26.44 |
|        8192 |      32.0 |         13.05 |      16.98 (+30%) |           31.63 |               31.44 |
|   **14336** |  **56.0** |     **16.08** |    **20.02 (+25%)** |       **35.88** |           **35.88** |
|       28672 |     112.0 |         19.00 |      21.28 (+12%) |           39.72 |               39.68 |

Units: GB/s, average across the 56 non-diagonal (src, dst) pairs.
`chunk_bytes = 2 * hidden` in the real fused-dispatch path; `14336` is
the DeepSeek-V3 default.

Key observations at the production chunk size (14336 B):

* **mori_shmem is ~2.2× faster than rocshmem** (35.9 vs 16.1 GB/s
  blocking). rocshmem's blocking put serializes through the IPC path
  with no P2P fast lane; mori_shmem exposes both an XGMI-direct lane
  (~42 GB/s to the nearest peer) and a baseline (~35 GB/s elsewhere).
* **On rocshmem, switching `putmem_warp` → `putmem_nbi_warp` + outer
  `fence()` gives +25% bandwidth essentially for free.** The blocking
  variant waits per-warp for completion; the NBI variant batches
  s_waitcnt drain into the single end-of-kernel fence
  (`rocshmem_fence_wave_wrapper`).
* **On mori_shmem the two variants are within noise.** The blocking
  impl already saturates the IPC P2P path, so the source-reuse
  guarantee comes for free and `putmem_warp` is the simpler choice.

### Warp-count sweep at the production chunk size (14336 B, 56 MiB total)

Same setup, varying `--num-warps` (= put-issuing warps per CTA), 8 SMs.

| backend    | num_warps | putmem_warp | putmem_nbi_warp |
| ---------- | --------: | ----------: | --------------: |
| mori_shmem |         4 |       35.88 |           35.88 |
| mori_shmem |     **8** |   **52.95** |           52.87 |
| mori_shmem |        16 |       54.64 |           54.43 |
| rocshmem   |         4 |       16.08 |           20.02 |
| rocshmem   |         8 |       28.58 |           30.29 |
| rocshmem   |    **16** |   **47.49** |       **50.14** |

Units: GB/s, average across the 56 non-diagonal (src, dst) pairs.

Implications for `ep_all2all_fused.py` (currently uses `num_warps=4`):

* **rocshmem scales near-linearly with warp count** (4→16 ≈ 3×): the
  4-warp deficit was issue-rate bound, not bandwidth bound. Pushing to
  16 warps closes most of the gap with mori_shmem (50 vs 55 GB/s).
* **mori_shmem saturates at 8 warps** (~53 GB/s), <4% headroom going
  to 16. 8 is the sweet spot.
* **The NBI advantage on rocshmem shrinks as warps grow** (+25% at 4
  warps → +6% at 16). Once enough warps are in flight, per-warp
  completion waits stop being the bottleneck. At low warp counts NBI
  is still meaningfully better.

### Reproduce

```bash
# rocshmem
ulimit -l unlimited
ROCSHMEM_HEAP_SIZE=2147483648 TRITON_DIST_SHMEM_BACKEND=rocshmem \
    bash ./scripts/launch_amd.sh \
         ./python/triton_dist/test/amd/test_putmem_warp_bw.py --size-sweep

# mori_shmem
TRITON_DIST_SHMEM_BACKEND=mori_shmem \
    bash ./scripts/launch_amd.sh \
         ./python/triton_dist/test/amd/test_putmem_warp_bw.py --size-sweep
```
