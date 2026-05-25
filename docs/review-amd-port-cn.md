# 代码评审：Triton-distributed AMD 移植

**分支：** `dev/port-amd`
**评审范围：** `c250ad0..575371c`（领先 `origin/main` 共 3 个提交）

- `575371c` — amd: reorder barrier_all_on_stream signature
- `4491131` — function: add amd stub package
- `c250ad0` — port to amd

## 评审方法

并行派出 5 个评审子代理，分别覆盖：浅层 bug 扫描、NVIDIA/AMD 对照差异、Git 历史上下文、源码注释合规性、HIP/ROCm 正确性。所有结论都回到源码逐一核实——一部分子代理标出的"高危问题"经过比对后发现是 **直接照搬自 NVIDIA 现有实现** 的模式，并非本分支引入的回归，已过滤掉。具体而言，以下问题在 `origin/main` 上已经存在，**不算回归**：

- `barrier_all_on_stream` 在 intra-node 快路径上没有把 `stream=` 传给 Triton kernel 启动
- `BarrierAllContext.symm_barrier` 被分配为 `(num_local_ranks,)`，而 `barrier_all_intra_node_non_atomic_block` 的 docstring 要求 `2*num_local_ranks` 槽位
- `barrier_all_intra_node_atomic_cas_block` 的两个阶段之间没有 `__syncthreads()`

参照：`python/triton_dist/kernels/nvidia/common_ops.py:154-242`。

---

## 本分支引入的真实问题

### 1. `scripts/launch_amd.sh` —— rocSHMEM/MPI 的 `LD_LIBRARY_PATH` 导出被整体注释掉

提交 `c250ad0` 把整段按后端区分的 `LD_LIBRARY_PATH` 导出代码注释掉了：

```bash
# if [ "${TRITON_DIST_SHMEM_BACKEND}" != "mori_shmem" ]; then
#   export LD_LIBRARY_PATH=${MPI_ROOT}/lib:${ROCSHMEM_ROOT}/lib:$LD_LIBRARY_PATH
# fi
```

而本文件第 26 行设置的默认后端是 `ROCSHMEM_BACKEND=IPC`——也就是 rocSHMEM 路径。除非容器镜像把 `librocshmem.so` / `libmpi.so` 放在了默认的 loader 查找路径上，否则启动时就会 `dlopen` 失败。原先 `if` 守卫的目的只是排除 `mori_shmem` 后端，恰好正是在保护本 PR 重点扩展的 rocSHMEM 用户。

**位置：** `scripts/launch_amd.sh:11-13`

**建议修复：** 恢复原先按后端守卫的导出语句；或在文档中明确说明新的容器镜像前置条件。

---

### 2. rocshmem 构建链路中的 MPI 安装路径不一致

`build_rocshmem.sh` 现在硬编码了 `MPI_ROOT=/opt/ompi`、`UCX_ROOT=/opt/ucx`，`shmem/rocshmem_bind/pyrocshmem/setup.py` 也同步改成 `/opt/ompi`。但是上层编排脚本 `shmem/rocshmem_bind/build.sh`（本 PR 没改）仍然在用：

```bash
export OPENMPI_UCX_INSTALL_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"
export PATH="${OPENMPI_UCX_INSTALL_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${OPENMPI_UCX_INSTALL_DIR}/lib:$LD_LIBRARY_PATH"
# 并把 -DOMPI_DIR=${OPENMPI_UCX_INSTALL_DIR} 传给 pyrocshmem 的 cmake
```

本 PR 之前三处路径是一致的（`/opt/ompi_build/install/ompi`）。改完之后，`build.sh` 仍然把 `/opt/ompi_build/install/ompi` 喂给 PATH/LD/cmake，而子脚本和 `pyrocshmem/setup.py` 已经看 `/opt/ompi` 了——三处不一致。

**位置：** `shmem/rocshmem_bind/build_rocshmem.sh:60-78` vs `shmem/rocshmem_bind/build.sh:50-55`

**建议修复：** 要么把 `build.sh` 也改成 `MPI_ROOT=/opt/ompi` 的新约定，要么把 `build_rocshmem.sh` 改回 `OMPI_INSTALL_DIR` 可被环境变量覆盖的写法，让三层保持一致。

---

### 3. `rocshmem_free_tensor_sync` 实际上没有释放内存

```python
def rocshmem_free_tensor_sync(tensor):
    """rocshmem symmetric tensors are freed when the Python ``SymmRocShmemBuffer``
    is garbage-collected. We synchronize so pending GPU work is drained
    *before* the caller drops its last reference, matching the semantics of
    :func:`nvshmem_free_tensor_sync`.
    """
    torch.cuda.synchronize()
```

兄弟函数 `nvshmem_free_tensor_sync` 在两次同步之间调用了 `nvshmem.core.free_tensor(tensor)`；`mori_shmem_free_tensor_sync` 也调用了 `mori_shmem.mori_shmem_free_tensor(tensor)`。新加的 rocshmem 版本接收了 `tensor` 参数但根本没用，只做了一次同步。docstring 里"matching the semantics of nvshmem_free_tensor_sync"的说法是不成立的——它依赖 Python GC 去触发 `SymmRocShmemBuffer` 的析构，而只要有任何调用方还持有引用（比如新加的 `ShmemLazyAllocator` 的内部记录），析构就不会发生。`BarrierAllContext.finalize()` 一遍遍调用之后，以及未来的 EP-MoE 路径正式接进来之后，对称堆都会越积越多。

**位置：** `python/triton_dist/utils.py:326-332`

**建议修复：** 要么真正调用 `pyrocshmem` 的释放 API；要么改写 docstring 并在函数被实际调用时直接 `raise`，等到释放接口接通再放开。

---

### 4. `BarrierAllContext.local_rank` 推导方式在非连续 PE 编号下不安全

```python
self.rank = get_triton_dist_world().rank()
self.local_world_size = (
    get_triton_dist_local_world_size()
    or int(_os.environ.get("LOCAL_WORLD_SIZE", "0"))
    or int(_os.environ.get("WORLD_SIZE", "1"))
)
self.local_rank = self.rank % self.local_world_size
```

NVIDIA 用 `pynvshmem.team_my_pe(TEAM_NODE)` 来取 `local_rank`——这是 SHMEM 节点团队里权威的位置。而本 PR 用 `rank % local_world_size` 替代，**只有当全局 PE 编号在每个节点内是连续的、并且每个节点都从 PE 0 起始时才是等价的**。一旦排布方式不同（rocSHMEM/mori 在某些拓扑下不保证），`barrier_all_intra_node_atomic_cas_block`（213-219 行）里 `local_rank_offset = rank - local_rank` 算出来的对端 flag 槽位就是错的，且是 **静默错误**，不会报错。

更严重的是：链式 `or` 兜底 `int(_os.environ.get("WORLD_SIZE", "1"))` 会在 multi-node 场景下（`LOCAL_WORLD_SIZE` 未设置）命中，结果就是 `local_world_size = WORLD_SIZE`（比如 2 节点 ×8 卡场景下值为 16）。再加上 `symm_barrier` 只按 `(num_local_ranks,)` 分配，intra-node CAS kernel 会拿着超大的索引去访问偏小的对称缓冲区——**对称堆越界访问**。

**位置：** `python/triton_dist/kernels/amd/common_ops.py:281-295`

**建议修复：** 若 rocSHMEM/mori 有 node-team 查询接口就走那个；没有的话，在 `LOCAL_WORLD_SIZE` 和 `get_triton_dist_local_world_size()` 都没有的场景下直接 `raise`，而不是悄悄退化到 `WORLD_SIZE`。

---

### 5. AMD 模块里 `NVSHMEM_SIGNAL_DTYPE` 被遮蔽成了 `uint64`

```python
# python/triton_dist/kernels/amd/common_ops.py
NVSHMEM_SIGNAL_DTYPE = MORI_SHMEM_SIGNAL_DTYPE  # = torch.uint64
```

权威定义在 `python/triton_dist/utils.py:661`：`NVSHMEM_SIGNAL_DTYPE = torch.int64`。NVIDIA 端 `python/triton_dist/kernels/nvidia/common_ops.py:375,397` 用它做 `signal_tensor.dtype` 的分派。AMD 端这里在模块作用域把同名符号重绑成了 `torch.uint64`，注释解释是为了让"downstream layer code that imports NVSHMEM_SIGNAL_DTYPE keeps working without rename"。但任何 `from triton_dist.kernels.amd.common_ops import NVSHMEM_SIGNAL_DTYPE` 然后跟 `signal_tensor.dtype` 比较（即 NVIDIA 的模式）或者做 `.view(torch.int64)` 的代码，都会 **静默走错分支**。

**位置：** `python/triton_dist/kernels/amd/common_ops.py:55-60`

**建议修复：** 把别名改为 `torch.int64`；或者干脆显式重命名，不要去遮蔽权威符号。

---

## 值得复核（置信度稍低）

### A. `get_moe_optim_config` 实际并非"逐字移植"

`python/triton_dist/function/amd/common.py` 在 `4491131` 的提交信息里被描述为"ported verbatim"，但 `get_moe_optim_config` 实际加了 `min(80, max_sms)`、`min(64, max_sms)` 两个 clamp，而 NVIDIA 没有；且把 NVIDIA 的 `max_sms > 78`（H800）/`<=78`（H20）分支合并掉了。在目前所有的 MI300/MI355 SKU（在某些分区模式下 CU 数小于 80/64）上，AMD 路径会静默地把 `num_dispatch_sms` / `num_combine_sms` 调小。确认一下是否是有意为之——如果是，把"verbatim"的提交信息修订一下，避免误导。

### B. `MORI_SIGNAL_SET` 在 import 时被一次性解析

`python/triton_dist/language/extra/libshmem_device.py:557-563` 根据 `is_rocshmem()` 在 `MORI_SIGNAL_SET = 0`（rocSHMEM）和 `9`（mori）之间切换，**但读取 `TRITON_DIST_SHMEM_BACKEND` 只在 import 那一刻发生**。如果有任何模块在该环境变量被设置之前先 import 了 `libshmem_device`，缓存的常量就是错的，对应的 `wait_until` 会死锁。再加上 `scripts/launch_amd.sh` 当前并不导出 `TRITON_DIST_SHMEM_BACKEND`，这是个真实存在的踩坑风险。建议要么改为懒解析，要么在首次使用时 assert 后端。

### C. `fence()` 改路由到了 `rocshmem_fence_wave_wrapper`

`python/triton_dist/language/extra/hip/librocshmem_device.py:433-452` 因为预编译 bitcode 里没有不带后缀的 `rocshmem_fence_wrapper`，把对外暴露的 `fence()` extern 改成调用 `_wave` 后缀版本。注释解释这两者"functionally equivalent"，因为内部都是 `rocshmem::rocshmem_fence()`。建议对照 rocSHMEM 的 collective 调用前置条件文档复核——`_wave` 后缀通常意味着需要整个 wavefront 集体进入，而像 `low_latency_all_to_all.py` 这样的调用方往往是从单线程发起 `fence()`。

---

## 已核实属于有意/正确的部分

- `575371c` 的 `barrier_all_on_stream` 签名调整：正确恢复了 `stream` 作为首个位置参数，让 `test_distributed-notify-wait.py` 这种历史调用方继续可用；两处 `gemm_reduce_scatter` 调用点也相应简化。
- `kernels/amd/ep_a2a.py` 的 `kernel_get_dispatch_send_reqs` 函数体跟 NVIDIA 版本字节对齐。
- `jit.py` 的修改有 `TRITON_DIST_DEBUG_ROCSHMEM_CTX` 环境变量守卫，对 NVIDIA 路径无影响。
- `setup.py` 的 `ROCM_ARCH` 改动可被环境变量覆盖，默认 `gfx942`；NVIDIA 构建路径未受影响。
- `language_extra.py` 新加的 int8/uint8/int16 `ld`/`st` 重载是纯增量，且匹配 NVIDIA 侧的接口形状。
- `librocshmem_device.py` 的 `_block` / `_warp` 别名正确地把 NVSHMEM 命名约定映射到了 rocSHMEM 的 `_wg` / `_wave` 原语。
- `function/amd/__init__.py` 和 `function/amd/ep_moe_fused.py` 作为存根正确地 `raise NotImplementedError`。

---

## 小结

5 个真实问题，都可以小范围修复，不需要动整体的移植结构。其中：

- **#1、#2（构建/启动脚本）**最容易立刻被踩到，从干净的容器跑第一次时就会以 `dlopen` 失败或编译路径错误的形式暴露出来。
- **#4（BarrierAllContext）** 和 **#5（NVSHMEM_SIGNAL_DTYPE）** 是静默数据正确性隐患，要等 AMD fused EP-MoE backend 接入、真正走到这两条路径时才会显式爆出来。
- **#3（rocshmem_free_tensor_sync）** 是对称堆泄漏，进程跑得越久越严重。
