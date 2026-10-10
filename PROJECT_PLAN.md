# Cache and Bandwidth Contention in Shared-Memory CPU–Accelerator SoCs

**CPE 5300 Course Project — Implementation Plan (Tiers 1–4)**

## Research question

In a shared-memory SoC where CPU cores and a systolic-array accelerator compete for the last-level cache (LLC) and DRAM bandwidth, how much contention can be recovered by **scheduling** (where/when kernels run) versus **hardware partitioning** (LLC way-partitioning), and how does the answer change with accelerator size and memory configuration? The contention-aware scheduler is framed as an **on-chip hardware dispatch unit**, implemented in SystemVerilog and synthesized for area/timing.

Project type (syllabus): *compare existing ideas that have not been directly compared* + *initial study of a new idea* (the dispatch unit).

## Tier overview

| Tier | Scope | Experiments enabled |
|---|---|---|
| **1 — Core** | Event engine, cores, phase-based systolic accelerator, DRAM bandwidth contention (grounded in C measurements), baseline policies (FIFO, HEFT, Qilin, Xu 2023, HaX-CoNN-style) + contention-aware policy, workloads, metrics, sweeps | E1 characterization, E2 scheduling-only, E5 design-space sweep |
| **2 — Shared LLC** | Capacity-based LLC model with miss-rate curves, way-partitioning (static + utility-based), contention tracking as an alternative to partitioning, optional bandwidth throttling, LLC→DRAM coupling | E3 memory-side only, E4 scheduling × memory-side |
| **3 — Energy + HW dispatch** | Energy model; contention-aware policy recast as a hardware dispatch unit built from published hardware schedulers (HW HEFT_RT, HTS) with MoCA-style bandwidth counters (fixed-point golden model, bounded tables, decision latency, analytical cost) | E6 energy; dispatch-latency sensitivity |
| **4 — RTL (stretch)** | SystemVerilog dispatch unit, co-simulation against Python golden model, Yosys synthesis, compared against HW HEFT_RT | Area/timing table vs. HW HEFT_RT; measured latency fed back into simulator |

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
  kernels.py           # Kernel, make_gmm_kernel, make_simple_kernel
  cores.py             # Core, CPU, Accelerator (standalone time)
  systolic.py          # weight-stationary systolic timing + per-phase traffic
  memory.py            # DRAM bandwidth allocation (Tier 1), LLC model (Tier 2)
  model.py             # rate model: Segment, Job
  simulator.py         # event loop: advance -> change -> reallocate, policy hooks
  config.py            # hardware/system configuration dataclasses
  energy.py            # Tier 3
  policies/
    base.py                  # Policy interface (on_queued, pick)
    fifo.py
    heft_greedy.py           # StarPU-style
    qilin_style.py
    xu2023.py                # recent same-problem baseline (1.8)
    haxconn_style.py         # HaX-CoNN cost model + PCCS-style slowdown (1.8)
    contention_aware.py      # software form (Tier 1)
    dispatch_unit_model.py   # fixed-point hardware golden model (Tier 3)
  partition/
    static.py
    ucp.py                   # utility-based LLC partitioning (Tier 2)
    tracking.py              # contention tracking instead of partitioning (Tier 2)
    throttle.py              # MoCA-style bandwidth throttling (Tier 2, optional)
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
  refs.bib
  notes/               # one reading note per paper (0.4)
```

---

## Phase 0 — Setup and framing

| Step | Task | Done when |
|---|---|---|
| 0.1 | Repo hygiene: `.gitignore`, `requirements.txt`, package `__init__.py`s, pytest config | `pytest` runs (0 tests) |
| 0.2 | `sim/config.py`: dataclasses for every hardware parameter (core count, core GFLOP/s, array rows/cols, clock, scratchpad size, LLC size/ways, DRAM BW, …) with documented defaults and sources | One place to change any parameter; no magic numbers elsewhere |
| 0.3 | Pick reference platform numbers (e.g., Apple M-series / AMD APU: LLC size, DRAM BW, core count) and cite them | Defaults table with citations |
| 0.4 | Read the related work below and write one note per paper (problem, mechanism, evaluation, how we differ, what we reuse) | Notes written; `paper/refs.bib` filled |
| 0.5 | Instructor check-in: confirm scope, tiers, and that the dispatch unit + RTL framing is acceptable | Notes recorded |

### 0.4 Reading list

| Paper | Venue | Reader | Used in |
|---|---|---|---|
| Efficient tasks scheduling in multicore systems integrated with hardware accelerators (Xu, Shi, Chen) | J. Supercomputing 2023 | you | 1.8 `xu2023.py` baseline; task model for 1.9 |
| PCCS: Processor-Centric Contention-aware Slowdown Model for Heterogeneous System-on-Chips (Xu, Belviranli, Shen, Vetter) | MICRO 2021 | you | 1.7 measurements; 1.8 `haxconn_style.py` (its contention model); E1 comparison |
| Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips, "HaX-CoNN" (Dagli, Belviranli) | PPoPP 2024 | you | 1.8 `haxconn_style.py` baseline; 1.9 layer dependencies; 1.10 metrics |
| A Hardware-based HEFT Scheduler Implementation for Dynamic Workloads on Heterogeneous SoCs (Fusco, Hassan, Mack, Akoglu) | VLSI-SoC 2022 | you | 3.2 template; 3.4 cost metrics; 4.4 comparison point (9.144 ns) |
| HTS: A Hardware Task Scheduler for Heterogeneous Systems | arXiv 2019 | you | 3.2 queue and select logic; 4.1 module design |
| MoCA: Memory-Centric, Adaptive Execution for Multi-Tenant Deep Neural Networks (Kim et al.) | HPCA 2023 | partner | 2.5 optional throttling; 3.2 bandwidth monitors; 4.1 `bw_monitor.sv` |
| Hardware support for contention tracking in CPU and GPU last-level cache | JSA 2025 | partner | 2.4 tracking vs. partitioning; 3.4 cost reference |
| Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference | 2025 | partner | 1.7 measurement method; 2.2/2.3 parameter justification |
| A fully hardware-managed scheduling architecture for AI accelerators | Springer 2026 | partner | 3.2 dependency tracking (if 1.9 adds dependencies); 4.1 module design |
| Qureshi & Patt, Utility-Based Cache Partitioning (UCP) | MICRO 2006 | partner | 2.4 dynamic partitioning |

Reading order for you: Xu 2023, PCCS, HaX-CoNN (needed before 1.8 baselines and 1.9), then HW HEFT and HTS (needed before Tier 3).

---

## Phase 1 — Tier 1: Core simulator

### 1.1 Event engine (`sim/engine.py`)
- Events: `(timestamp, tiebreaker, event_type, data)` on a `heapq`; tiebreaker = monotonic counter.
- `register(type, handler)`, `schedule(t, type, data)`, `run(until=None)`.
- **Cancellable events** (needed by 1.4): each scheduled event gets an id/version; cancelled or stale events are skipped when popped (lazy deletion).
- Reject scheduling in the past.
- **Validate:** out-of-order events fire in time order; ties fire in schedule order; `data` never compared; cancelled events never fire; handler-scheduled events work.

### 1.2 Kernels and tasks (`sim/kernels.py`)
- `Kernel`: id, type (`GEMM`, `CPU_ONLY`, …), dims (M, K, N), arrival time, bytes read/written, working-set size (used in Tier 2), parallelizable flag.
- Runtime fields: assigned resource, start, finish, remaining work.
- **Validate:** byte/FLOP counts match hand calculation for a few shapes.

### 1.3 Resources (`sim/cores.py`)
- `Core`: id, busy/idle, current task; `can_run`, `assign`, `release`. Optional multi-core span with fixed speedup factor.
- `Accelerator`: one kernel at a time + waiting deque.
- **Validate:** trivial "always Core 0" rule — busy→idle transitions and timestamps correct for 5 hand-traced kernels.

### 1.4 Rate-based execution model (`sim/model.py`, `sim/simulator.py`, key design piece)
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
- Co-run: CPU STREAM concurrently with accelerator matmul load (MPS/Accelerate); record standalone vs. co-run bandwidth and matmul time. Follow the measurement method of the mobile SoC characterization paper.
- Single-thread vs. all-core bandwidth (one core cannot saturate DRAM), for a `max_bw_per_core` limit.
- Standalone bandwidth of individual kernel types (GEMM alone, memory-bound loop alone): the input a PCCS-style slowdown model needs.
- Write `data/bandwidth_measurements.csv`; fit degradation curve (scipy) in Python.
- **Validate:** STREAM numbers stable across runs (report mean ± std); fitted curve plotted with residuals.

### 1.8 Scheduling policies (`sim/policies/`)
- `base.py`: common interface `on_arrival`, `on_resource_free`, `choose(task, state)`.
- `fifo.py`: arrival order, first free resource.
- `heft_greedy.py`: StarPU-style earliest-finish-time using standalone performance models (no contention awareness).
- `qilin_style.py`: per-resource curve fit on execution history, adaptive split.
- `contention_aware.py` (software form): predicted finish time *including* current bandwidth contention + history-based correction term.
- `xu2023.py`: recent same-problem baseline (multicore + accelerators, minimize total execution time, no memory contention). Implemented from the paper; docstring lists every simplification.
- `haxconn_style.py`: HaX-CoNN's contention cost model with a PCCS-style slowdown estimate, run inside the online loop (the full SAT/Z3 solver is out of scope; state this).
- **Validate:** each policy's decisions on a 5-kernel hand-traced workload match expected assignments.

### 1.9 Workloads (`sim/workload.py`)
- Real layer shapes (e.g., ResNet-50, BERT-base, MobileNet) in `data/layer_shapes/`; CPU-side kernels (pre/post-processing, memory-bound ops).
- Mixes: accelerator-heavy / balanced / CPU-heavy × low / medium / high contention (arrival rate, sizes).
- Seeded RNG; identical workloads across policies.
- Decision (after reading Xu 2023 and HaX-CoNN): keep kernels independent, or add chain dependencies (each DNN is a chain of layers; a layer is released when the previous one finishes). Chains only, not general task graphs.
- **Validate:** same seed → identical trace; summary stats per mix.

### 1.10 Metrics (`sim/metrics.py`)
- Makespan (cycles), throughput (kernels or inferences per second, comparable to HaX-CoNN), per-kernel latency (mean, p95, p99), core/accelerator utilization, DRAM bandwidth utilization over time, co-run slowdown.
- **Validate:** metrics on the hand-traced workload match hand values.

### 1.11 Experiment harness (`experiments/`)
- Config-driven runner, parameter sweeps, CSV results in `data/results/`, plotting scripts.
- **E1** characterization: co-run slowdown vs. array size, scratchpad size, DRAM BW; compare the simulator's slowdown against a PCCS-style prediction.
- **E2** scheduling-only: 6 policies (FIFO, HEFT, Qilin, Xu 2023, HaX-CoNN-style, contention-aware) × workload mixes.
- **E5** design-space sweep: best policy across array sizes (16², 32², 64²) and bandwidth levels.
- **Tier 1 exit:** E1, E2, E5 figures produced; first paper draft outline.

---

## Phase 2 — Tier 2: Shared LLC and partitioning

### 2.1 Miss-rate curves
- Per-kernel miss-rate-vs-capacity curve (from working-set size / reuse model; e.g., analytical for GEMM tiling, simple power-law for CPU kernels).
- **Validate:** curves monotone non-increasing; miss rate ≈ 0 when capacity ≥ working set.

### 2.2 Shared LLC model (`sim/memory.py`)
- Capacity-based (no address-level simulation): total ways, way size; unpartitioned sharing approximated by occupancy proportional to access rate.
- The LLC represents the SoC-level cache shared by the CPU and the accelerator (not the CPU cluster's private L2); justify sizes and bandwidth with the mobile SoC characterization paper.
- Way-partitioning: CPU vs. accelerator way allocation.
- **Validate:** full allocation to one side reproduces its standalone miss rate.

### 2.3 LLC → DRAM coupling
- DRAM demand = access rate × miss rate; feeds the bandwidth model from 1.6 and the rate updates in 1.4.
- **Validate:** shrinking the LLC raises DRAM traffic and lengthens memory-bound kernels as expected.

### 2.4 Partitioning policies (`sim/partition/`)
- Static splits (e.g., 25/50/75% to accelerator).
- Dynamic utility-based (UCP-style) repartitioning at fixed intervals.
- Contention tracking instead of partitioning (demotion counters, JSA 2025): implement as an option, or at least discuss, since it argues strict partitioning loses throughput.
- **Validate:** UCP converges to the better static split on a stable workload.

### 2.5 Bandwidth throttling (optional, MoCA-style)
- Per-agent bandwidth caps set by a simple runtime rule, as a third memory-side approach next to partitioning and tracking.
- First thing to drop if time runs short.
- **Validate:** caps respected in the bandwidth log; capped agent slows as predicted.

### 2.6 Experiments
- **E3** memory-side only (FIFO + each partition scheme, tracking, and throttling if implemented).
- **E4** scheduling × memory-side grid: additive, overlapping, or conflicting?
- Extend E5 with LLC size.
- **Tier 2 exit:** E3/E4 figures; paper draft with full evaluation section.

---

## Phase 3 — Tier 3: Energy and hardware dispatch unit

### 3.1 Energy model (`sim/energy.py`)
- Per-op energies (MAC, SRAM/LLC access, DRAM access per byte) and active/idle power for cores and accelerator; cite sources (e.g., Horowitz ISSCC 2014, Eyeriss/Accelergy tables).
- **Validate:** energy breakdown for a single kernel matches hand calculation.

### 3.2 Dispatch unit microarchitecture spec
- Built from published hardware schedulers rather than from scratch: HEFT-style finish-time tracking from HW HEFT_RT (Fusco et al.), queue and ready/select logic from HTS, per-agent bandwidth counters from MoCA, dependency tracking from the Springer 2026 scheduler (only if 1.9 adds dependencies).
- Inputs: bandwidth counters (per-agent bytes/window), resource busy flags, kernel descriptors. Start with DRAM bandwidth counters only; LLC contention counters are optional.
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
- Report in the same form as HW HEFT_RT (decision latency, resource cost) so the two can be compared directly; LLC contention-tracking counters (~0.66% area) as a reference point.

### 3.5 Experiments
- **E6** energy across policies × partitioning; do makespan and energy rankings agree?
- Dispatch-latency sensitivity (1, 4, 16, 64 cycles) and table-size sensitivity, with HW HEFT_RT's 9.144 ns as the reference value.
- **Tier 3 exit:** energy figures + hardware cost table in draft.

---

## Phase 4 — Tier 4 (stretch): SystemVerilog dispatch unit

### 4.1 RTL modules (`rtl/`)
- `kernel_queue.sv`, `history_table.sv`, `bw_monitor.sv`, `dispatch_unit.sv` (top: estimator + select FSM).
- Synthesizable SV only; parameterized by queue depth, table size, bit widths.
- Design references: HTS for the queue and select logic, MoCA for `bw_monitor.sv` (counter width, window, reset), Springer 2026 for dependency tracking if needed.
- This tier produces the direct hardware comparison against HW HEFT_RT, so protect time for it.

### 4.2 Unit testbenches
- Per-module directed tests (queue full/empty, table hit/miss/evict, counter windows).

### 4.3 Co-simulation against golden model
- Verilator testbench replays decision traces from 3.3; outputs must match **bit-exactly**.
- **Validate:** 0 mismatches over traces from every workload mix.

### 4.4 Synthesis
- Yosys synthesis (generic cells, plus Nangate45/ASAP7 liberty if available); OpenSTA timing.
- Report cell count/area, critical path, achievable frequency; sweep queue/table sizes.
- Compare against HW HEFT_RT (9.144 ns average decision latency): comparable latency, plus contention awareness, for how much extra area.

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
| After Tier 4 | RTL implementation, area/timing vs. HW HEFT_RT |
| End | Conclusions, limitations, AI-use disclosure, presentation slides |


## Working rules

- One step at a time; each step has a validation test that must pass before moving on.
- All hardware parameters live in `sim/config.py` with cited sources.
- All experiments seeded and reproducible from a config file.
- Commit after each validated step.
