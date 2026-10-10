# Cache and Bandwidth Contention in Shared-Memory CPU–Accelerator SoCs

**CPE 5300 Course Project — Implementation Plan (Tiers 1–4)**

## Research question

In a shared-memory SoC where CPU cores and a systolic-array accelerator compete for the last-level cache (LLC) and DRAM bandwidth, how much contention can be recovered by **scheduling** (where/when kernels run) versus **hardware partitioning** (LLC way-partitioning), and how does the answer change with accelerator size and memory configuration? The contention-aware scheduler is framed as an **on-chip hardware dispatch unit**, implemented in SystemVerilog and synthesized for area/timing.

Project type (syllabus): *compare existing ideas that have not been directly compared* + *initial study of a new idea* (the dispatch unit).

## Tier overview

| Tier | Scope | Experiments enabled |
|---|---|---|
| **1 — Core** | Event engine, cores, phase-based systolic accelerator, DRAM bandwidth contention (grounded in C measurements), baseline + contention-aware policies, workloads, metrics, sweeps | E1 characterization, E2 scheduling-only, E5 design-space sweep |
| **2 — Shared LLC** | Capacity-based LLC model with miss-rate curves, way-partitioning (static + utility-based), LLC→DRAM coupling | E3 partitioning-only, E4 scheduling × partitioning |
| **3 — Energy + HW dispatch** | Energy model; contention-aware policy recast as a hardware dispatch unit (fixed-point golden model, bounded tables, decision latency, analytical cost) | E6 energy; dispatch-latency sensitivity |
| **4 — RTL (stretch)** | SystemVerilog dispatch unit, co-simulation against Python golden model, Yosys synthesis | Area/timing table; measured latency fed back into simulator |

Each tier ends with a complete, writable paper. Never start a tier until the previous tier's validation passes.

## Languages and tools

| Part | Language / tool |
|---|---|
| Simulator, policies, workloads, metrics, experiments, plots | Python 3.13 (numpy, scipy, pandas, matplotlib, pytest) |
| Bandwidth + co-run measurements | C with OpenMP (STREAM-style); accelerator load via PyTorch MPS or Accelerate `cblas_sgemm` |
| Dispatch unit RTL | SystemVerilog; Verilator (sim + co-sim), Yosys (synthesis), OpenSTA (timing, optional) |

## Repository layout (target)

```
sim/
  engine.py            # discrete-event engine (cancellable events)
  tasks.py             # Kernel / Task
  resources.py         # Core, Accelerator
  memory.py            # DRAM bandwidth model (Tier 1), LLC model (Tier 2)
  systolic.py          # weight-stationary systolic timing + per-phase traffic
  energy.py            # Tier 3
  config.py            # hardware/system configuration dataclasses
  policies/
    base.py
    fifo.py
    heft_greedy.py
    qilin_style.py
    contention_aware.py      # software form (Tier 1)
    dispatch_unit_model.py   # fixed-point hardware golden model (Tier 3)
  partition/
    static.py
    ucp.py                   # utility-based LLC partitioning (Tier 2)
  workload.py
  metrics.py
bench/
  stream_corun.c
  Makefile
  run_measurements.sh
rtl/                   # Tier 4
  dispatch_unit.sv
  kernel_queue.sv
  history_table.sv
  bw_monitor.sv
  tb/
  synth/
experiments/
  configs/             # YAML/py configs per experiment
  run_experiment.py
  sweeps.py
  plots/
data/
  bandwidth_measurements.csv
  layer_shapes/        # real model layer dimensions
  results/
tests/
paper/
```

---

## Phase 0 — Setup and framing

| Step | Task | Done when |
|---|---|---|
| 0.1 | Repo hygiene: `.gitignore`, `requirements.txt`, package `__init__.py`s, pytest config | `pytest` runs (0 tests) |
| 0.2 | `sim/config.py`: dataclasses for every hardware parameter (core count, core GFLOP/s, array rows/cols, clock, scratchpad size, LLC size/ways, DRAM BW, …) with documented defaults and sources | One place to change any parameter; no magic numbers elsewhere |
| 0.3 | Pick reference platform numbers (e.g., Apple M-series / AMD APU: LLC size, DRAM BW, core count) and cite them | Defaults table with citations |
| 0.4 | Add related work for new scope: LLC partitioning (Qureshi & Patt, UCP, MICRO 2006), CPU–GPU/accelerator memory interference, SCALE-Sim | Reading list in `paper/refs.bib` |
| 0.5 | Instructor check-in: confirm scope, tiers, and that the dispatch unit + RTL framing is acceptable | Notes recorded |

---

## Phase 1 — Tier 1: Core simulator

### 1.1 Event engine (`sim/engine.py`)
- Events: `(timestamp, tiebreaker, event_type, data)` on a `heapq`; tiebreaker = monotonic counter.
- `register(type, handler)`, `schedule(t, type, data)`, `run(until=None)`.
- **Cancellable events** (needed by 1.4): each scheduled event gets an id/version; cancelled or stale events are skipped when popped (lazy deletion).
- Reject scheduling in the past.
- **Validate:** out-of-order events fire in time order; ties fire in schedule order; `data` never compared; cancelled events never fire; handler-scheduled events work.

### 1.2 Kernels and tasks (`sim/tasks.py`)
- `Kernel`: id, type (`GEMM`, `CPU_ONLY`, …), dims (M, K, N), arrival time, bytes read/written, working-set size (used in Tier 2), parallelizable flag.
- Runtime fields: assigned resource, start, finish, remaining work.
- **Validate:** byte/FLOP counts match hand calculation for a few shapes.

### 1.3 Resources (`sim/resources.py`)
- `Core`: id, busy/idle, current task; `can_run`, `assign`, `release`. Optional multi-core span with fixed speedup factor.
- `Accelerator`: one kernel at a time + waiting deque.
- **Validate:** trivial "always Core 0" rule — busy→idle transitions and timestamps correct for 5 hand-traced kernels.

### 1.4 Rate-based execution model (key design piece)
- Each running kernel has **remaining work** and a **current rate**. Whenever the set of running kernels changes (any start/finish/phase change), recompute every kernel's rate from available bandwidth, advance progress, cancel and reschedule its FINISH event.
- This is what makes contention *dynamic* rather than a fixed slowdown applied at start.
- **Validate:** two identical memory-bound kernels sharing bandwidth each take 2× standalone; when one finishes, the other speeds up — check hand-computed finish times.

### 1.5 Systolic accelerator timing (`sim/systolic.py`)
- Weight-stationary, `rows × cols` PEs; `tiles_k = ceil(K/rows)`, `tiles_n = ceil(N/cols)`.
- Per tile: **weight-load phase** (bandwidth-heavy: `rows*cols*bytes` from memory) and **compute phase** (`M + rows + cols` cycles, input streaming + fill/drain).
- Scratchpad size parameter: determines whether input activations are reused across tiles or re-fetched.
- Option for double-buffered weights (overlap load/compute).
- Emits a per-phase **bandwidth demand profile** consumed by 1.4.
- **Validate:** cycle counts match hand calculation; sanity-check against SCALE-Sim numbers for 2–3 layer shapes.

### 1.6 DRAM bandwidth model (`sim/memory.py`)
- Shared pool with peak bandwidth; allocation among concurrent demanders (proportional-share baseline), shaped by the measured degradation curve from 1.7 (effective bandwidth drops under co-run beyond simple sharing).
- **Validate:** total allocated ≤ peak; single-agent case returns standalone bandwidth; curve reproduces measured points.

### 1.7 Real-hardware measurements (`bench/`, C)
- STREAM-style copy/scale/add/triad with OpenMP, thread count sweep.
- Co-run: CPU STREAM concurrently with accelerator matmul load (MPS/Accelerate); record standalone vs. co-run bandwidth and matmul time.
- Write `data/bandwidth_measurements.csv`; fit degradation curve (scipy) in Python.
- **Validate:** STREAM numbers stable across runs (report mean ± std); fitted curve plotted with residuals.

### 1.8 Scheduling policies (`sim/policies/`)
- `base.py`: common interface `on_arrival`, `on_resource_free`, `choose(task, state)`.
- `fifo.py`: arrival order, first free resource.
- `heft_greedy.py`: StarPU-style earliest-finish-time using standalone performance models (no contention awareness).
- `qilin_style.py`: per-resource curve fit on execution history, adaptive split.
- `contention_aware.py` (software form): predicted finish time *including* current bandwidth contention + history-based correction term.
- **Validate:** each policy's decisions on a 5-kernel hand-traced workload match expected assignments.

### 1.9 Workloads (`sim/workload.py`)
- Real layer shapes (e.g., ResNet-50, BERT-base, MobileNet) in `data/layer_shapes/`; CPU-side kernels (pre/post-processing, memory-bound ops).
- Mixes: accelerator-heavy / balanced / CPU-heavy × low / medium / high contention (arrival rate, sizes).
- Seeded RNG; identical workloads across policies.
- **Validate:** same seed → identical trace; summary stats per mix.

### 1.10 Metrics (`sim/metrics.py`)
- Makespan (cycles), per-kernel latency (mean, p95, p99), core/accelerator utilization, DRAM bandwidth utilization over time, co-run slowdown.
- **Validate:** metrics on the hand-traced workload match hand values.

### 1.11 Experiment harness (`experiments/`)
- Config-driven runner, parameter sweeps, CSV results in `data/results/`, plotting scripts.
- **E1** characterization: co-run slowdown vs. array size, scratchpad size, DRAM BW.
- **E2** scheduling-only: 4 policies × workload mixes.
- **E5** design-space sweep: best policy across array sizes (16², 32², 64²) and bandwidth levels.
- **Tier 1 exit:** E1, E2, E5 figures produced; first paper draft outline.

---

## Phase 2 — Tier 2: Shared LLC and partitioning

### 2.1 Miss-rate curves
- Per-kernel miss-rate-vs-capacity curve (from working-set size / reuse model; e.g., analytical for GEMM tiling, simple power-law for CPU kernels).
- **Validate:** curves monotone non-increasing; miss rate ≈ 0 when capacity ≥ working set.

### 2.2 Shared LLC model (`sim/memory.py`)
- Capacity-based (no address-level simulation): total ways, way size; unpartitioned sharing approximated by occupancy proportional to access rate.
- Way-partitioning: CPU vs. accelerator way allocation.
- **Validate:** full allocation to one side reproduces its standalone miss rate.

### 2.3 LLC → DRAM coupling
- DRAM demand = access rate × miss rate; feeds the bandwidth model from 1.6 and the rate updates in 1.4.
- **Validate:** shrinking the LLC raises DRAM traffic and lengthens memory-bound kernels as expected.

### 2.4 Partitioning policies (`sim/partition/`)
- Static splits (e.g., 25/50/75% to accelerator).
- Dynamic utility-based (UCP-style) repartitioning at fixed intervals.
- **Validate:** UCP converges to the better static split on a stable workload.

### 2.5 Experiments
- **E3** partitioning-only (FIFO + each partition scheme).
- **E4** scheduling × partitioning grid: additive, overlapping, or conflicting?
- Extend E5 with LLC size.
- **Tier 2 exit:** E3/E4 figures; paper draft with full evaluation section.

---

## Phase 3 — Tier 3: Energy and hardware dispatch unit

### 3.1 Energy model (`sim/energy.py`)
- Per-op energies (MAC, SRAM/LLC access, DRAM access per byte) and active/idle power for cores and accelerator; cite sources (e.g., Horowitz ISSCC 2014, Eyeriss/Accelergy tables).
- **Validate:** energy breakdown for a single kernel matches hand calculation.

### 3.2 Dispatch unit microarchitecture spec
- Inputs: bandwidth counters (per-agent bytes/window), resource busy flags, kernel descriptors.
- Structures: N-entry kernel queue, M-entry history table (kernel class → observed slowdown), finish-time estimator, selection logic.
- Fixed-point formats and bit widths; decision latency in cycles.
- Block diagram for the paper.

### 3.3 Fixed-point golden model (`sim/policies/dispatch_unit_model.py`)
- Bit-accurate Python model of 3.2 (integer arithmetic only, bounded tables, eviction policy).
- Plugs into the simulator as a policy; dispatch latency charged per decision.
- Exports decision traces (inputs → outputs) for RTL co-simulation.
- **Validate:** results within a few % of the software `contention_aware` policy; document the gap from quantization/bounded tables.

### 3.4 Analytical hardware cost
- Storage bits, comparators/adders, estimated area (pre-RTL).

### 3.5 Experiments
- **E6** energy across policies × partitioning; do makespan and energy rankings agree?
- Dispatch-latency sensitivity (1, 4, 16, 64 cycles) and table-size sensitivity.
- **Tier 3 exit:** energy figures + hardware cost table in draft.

---

## Phase 4 — Tier 4 (stretch): SystemVerilog dispatch unit

### 4.1 RTL modules (`rtl/`)
- `kernel_queue.sv`, `history_table.sv`, `bw_monitor.sv`, `dispatch_unit.sv` (top: estimator + select FSM).
- Synthesizable SV only; parameterized by queue depth, table size, bit widths.

### 4.2 Unit testbenches
- Per-module directed tests (queue full/empty, table hit/miss/evict, counter windows).

### 4.3 Co-simulation against golden model
- Verilator testbench replays decision traces from 3.3; outputs must match **bit-exactly**.
- **Validate:** 0 mismatches over traces from every workload mix.

### 4.4 Synthesis
- Yosys synthesis (generic cells, plus Nangate45/ASAP7 liberty if available); OpenSTA timing.
- Report cell count/area, critical path, achievable frequency; sweep queue/table sizes.

### 4.5 Close the loop
- Feed measured decision latency (cycles at target clock) back into the simulator; re-run E2/E4 with it.
- **Tier 4 exit:** area/timing table; final evaluation uses RTL-derived latency.

---

## Phase 5 — Paper and presentation (continuous)

| When | Paper section |
|---|---|
| After Phase 0 | Introduction, motivation, related work |
| After Tier 1 | Methodology (simulator, systolic model, measurements), E1/E2/E5 |
| After Tier 2 | LLC model, E3/E4 |
| After Tier 3 | Dispatch unit design, energy, E6 |
| After Tier 4 | RTL implementation, area/timing |
| End | Conclusions, limitations, AI-use disclosure, presentation slides |


## Working rules

- One step at a time; each step has a validation test that must pass before moving on.
- All hardware parameters live in `sim/config.py` with cited sources.
- All experiments seeded and reproducible from a config file.
- Commit after each validated step.
