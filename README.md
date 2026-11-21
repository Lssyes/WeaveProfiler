# WeaveProfiler (`wp`)

Turnkey **Skill + MCP + Hook** GPU intra-kernel profiling and deterministic 4-gate verification infrastructure for **Sparse Attention / GDN / MoE Megakernels** across **NVIDIA H20 (`sm_90a`)**, **NVIDIA B300 (`sm_100a` / `sm_103a`)**, and **AMD Instinct MI308X (`gfx942`)** in **CUDA C++**, **HIP C++**, **CuTeDSL**, and **Triton**.

---

## 1. Quickstart for Any New User (1 Command)

```bash
git clone https://github.com/ziyuhuang123/WeaveProfiler.git

python3 -B WeaveProfiler/run.py install-agent \
  --workspace /path/to/your_kernel_dir \
  --protect kernel.cu \
  --card h20
```

Then enter your kernel directory and run **`claude`** or **`codex`** directly (or run headless supervised campaigns via `guard-exec`):

```bash
# Interactive Claude Code or Codex CLI (MCP + Hooks + Skill already wired):
cd /path/to/your_kernel_dir && claude
cd /path/to/your_kernel_dir && codex

# Non-interactive batch supervisor (auto-enables Linux bwrap RO-mount on WeaveProfiler when available):
python3 -B WeaveProfiler/run.py guard-exec \
  --workspace /path/to/your_kernel_dir \
  -- codex exec "Profile the baseline megakernel with weaveprof_megakernel_audit and eliminate SM straggler skew"
```

### What `install-agent` Provisions Automatically
| Layer | Claude Code | OpenAI Codex CLI | Guarantee |
| :--- | :--- | :--- | :--- |
| **MCP Server** (`weaveprof/mcp_server.py`) | `.mcp.json` + `.claude/settings.local.json` | `.codex/config.toml` (`[mcp_servers.weaveprofiler]`) | Exposes `weaveprof_status`, `weaveprof_megakernel_audit`, `weaveprof_profile`, `weaveprof_plan_context`, `weaveprof_analyze`, `weaveprof_compare_rounds`, `weaveprof_verify_seal` |
| **Hard Hooks** (`weaveprof/guard_hook.py`) | `.claude/settings.json` (`PreToolUse`, `PostToolUse`, `Stop`) | `.codex/hooks.json` + `hooks = true` | Blocks kernel edits before a `PASS` HMAC seal; auto-reverts unsealed `Bash` mutations from byte-exact backups; blocks probe macros in production files; blocks `Stop` until the final kernel is re-sealed |
| **Skill & Guide** | `.claude/skills/weaveprofiler/SKILL.md` + `CLAUDE.md` | `.agents/skills/weaveprofiler/SKILL.md` + `AGENTS.md` | Injects hardware invariant contracts & the supervised optimization loop |
| **Out-of-Tree Secret** | `~/.cache/weaveprof/sessions/<id>/.supervisor.key` | Same (`0600` permissions) | HMAC key lives outside the workspace (`Read`/`Bash` blocked by hook) so `seal.json` cannot be forged |

---

## 2. First-Class `wp` Recorder Frontends (Zero Kernel Signature Changes)

- **CUDA C++** (`2_instrument/recorders/cuda_cpp/wp_lite.cuh`):
  - Context modes: `-DWP_CTX_1R` (1 live 32-bit register + uniform `LDC` constant load, `0 B` SMEM), `-DWP_CTX_1R_NOINLINE`, `-DWP_SMEM_ZREG` (`16 B` static `__shared__` + `__noinline__`, `0` live loop registers).
  - Multi-file `.so` megakernels (`DSA-learn/mk`): add `-DWP_LEVEL=2 -DWP_CTX_1R -DWP_DEFINE_HOST_API` and drive from Python via `weaveprof.WpSoController(so_path)` (zero C++ boilerplate).
  - Runtime `.cubin` JIT kernels (`weave` / DeepGEMM `cudaLibrary_t`): call `wp_attach_library(library, &wp_buf)` — zero `__global__` parameter changes and zero extra `STO_ENTRY` symbols.
- **CuTeDSL** (`2_instrument/recorders/cutedsl/wp_cute.py`): `WpBuffer`, `wp_open`, `wp_b`, `wp_e`, `wp_close` with vectorized `<2 ms` `WPLITE02` serialization.
- **Triton** (`2_instrument/recorders/triton/wp_triton.py`, `wp_triton_amd.py`) & **HIP C++** (`2_instrument/recorders/hip_cpp/wp_lite_hip.h`).
- **Fast Decoder & Megakernel Audit** (`4_validate/tools/wp_decode.py`, `weaveprof/megakernel.py`):
  - Content-addressed `L0` build cache + parallel `L0`/`L_inst` compilation + paired `ABBA` warm timing;
  - Four hard fidelity gates: compiler spill/`STL`/`LDL` delta gate, exact/tolerance output gate, 0-instruction stub & pairing/overflow gate, and warm overhead gate ($\le 10\%$);
  - Millisecond cross-SM straggler & `grid_sync` barrier-starvation partitioning (`analyze_megakernel_skew`) + direct `sqlite3` 200 kHz `nsys` PMU extraction (`extract_nsys_sqlite_fast`).

---

## 3. Offline Verification Suite

```bash
python3 -B tests/package_structure.py
python3 -B -m unittest tests/test_mcp_hook_installer.py tests/test_wp_megakernel.py
bash tests/run_all.sh
```
