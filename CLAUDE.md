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
