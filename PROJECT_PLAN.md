# Contention-Aware Scheduling for Shared-Memory CPU–Accelerator SoCs

**CPE 5300 Course Project — Implementation Plan (Tiers 1–3)**

## Research question

In a shared-memory SoC where CPU cores and a systolic-array accelerator compete for DRAM bandwidth, can a scheduler that models this contention produce better schedules than existing heterogeneous schedulers that ignore it, and what would it cost to run as an **on-chip dispatch unit**? We compare FIFO, HEFT, Qilin, Xu 2023 (ESHCS) and a HaX-CoNN-style policy against a **new contention-aware algorithm** (dynamic-programming allocation + greedy dispatch + contention heuristics), measure how close each one gets to the optimal schedule, and estimate each algorithm's hardware cost. The answer is studied across accelerator sizes and memory bandwidths.

Project type (syllabus): *compare existing ideas that have not been directly compared* + *initial study of a new idea* (the contention-aware algorithm).

Goal metric: **schedule quality** (makespan and latency under contention) first; scheduling overhead (time per decision) second.

## Scope change (2026-11-09)

With about 5 weeks left, the scope is cut to the simulator, the new algorithm, and an analytical hardware cost:
- **Shared LLC and partitioning** (old Tier 2): moved to Future work (Phase 4).
- **SystemVerilog dispatch unit** (old Tier 4): moved to Future work (Phase 4), replaced by the analytical hardware cost in Tier 3.
- The architecture side of the project comes from the contention model (shared DRAM bandwidth, memory-bound vs. compute-bound systolic array), the design-space sweep (array size, scratchpad, bandwidth), and the hardware cost of each scheduler.

## Tier overview

| Tier | Scope | Experiments enabled |
|---|---|---|
| **1 — Core** | Event engine, cores, phase-based systolic accelerator, DRAM bandwidth contention (grounded in C measurements), task dependencies, DNN-chain and DAG workloads, baseline policies (FIFO, HEFT, Qilin, Xu 2023, HaX-CoNN-style) + contention-aware policy, optimal schedule and lower bounds, metrics, sweeps | E1 characterization, E2 scheduling comparison, E5 design-space sweep |
| **2 — New algorithm** | DP allocation + greedy dispatch + contention heuristics ("cost I cause others", compute/memory pairing), built and measured one piece at a time | E3 quality vs. baselines and optimal, E4 ablation, E5 extended |
| **3 — Hardware cost + energy** | Analytical hardware cost of every scheduler (operations per decision, state, decision latency, dispatch-unit feasibility) vs. HW HEFT_RT; dispatch-latency sensitivity; energy model (optional) | E6 overhead and hardware cost, E7 energy (optional) |

Each tier ends with a complete, writable paper. Never start a tier until the previous tier's validation passes.

## Schedule (5 weeks)

| Week | Work |
|---|---|
| 1 (Nov 9) | Dependencies in the simulator; 1.9 chain and DAG workloads; 1.10 metrics; start `xu2023.py` |
| 2 (Nov 16) | `xu2023.py`, `haxconn_style.py`, `optimal.py` + lower bounds; 1.11 harness; first E1/E2 runs; 1.7 measurements in parallel |
| 3 (Nov 23, short week) | Tier 2: cost model, DP allocation, greedy dispatch; first E3 results |
| 4 (Nov 30) | Tier 2 heuristics, E4 ablation, E5 sweep; Tier 3 hardware cost table |
| 5 (Dec 7) | Final experiments, paper, presentation |

## Languages and tools

| Part | Language / tool |
|---|---|
| Simulator, policies, workloads, metrics, experiments, plots | Python 3.13 (numpy, scipy, pandas, matplotlib, pytest) |
| Bandwidth + co-run measurements | C with OpenMP (STREAM-style); accelerator load via PyTorch MPS or Accelerate `cblas_sgemm` |

## Repository layout (target)

```
sim/
  engine.py            # discrete-event engine (cancellable events)
  kernels.py           # Kernel, make_gmm_kernel, make_simple_kernel
  cores.py             # Core, CPU, Accelerator (standalone time)
  systolic.py          # weight-stationary systolic timing + per-phase traffic
  memory.py            # DRAM bandwidth allocation
  model.py             # rate model: Segment, Job
  simulator.py         # event loop: advance -> change -> reallocate, policy hooks, dependencies
  config.py            # hardware/system configuration dataclasses
  energy.py            # Tier 3 (optional)
  hw_cost.py           # Tier 3: analytical hardware cost per scheduler
  policies/
    base.py                  # Policy interface (on_queued, pick)
    fifo.py
    heft_greedy.py           # StarPU-style
    qilin_style.py
    xu2023.py                # ESHCS, recent same-problem baseline (1.8)
    haxconn_style.py         # HaX-CoNN cost model + PCCS-style slowdown (1.8)
    contention_aware.py      # software form (Tier 1)
    optimal.py               # exact schedule + lower bounds on small DAGs (1.8)
    contention_dp.py         # new algorithm (Tier 2)
  workload.py
  metrics.py
bench/
  stream_corun.c
  Makefile
  run_measurements.sh
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
| 0.5 | Instructor check-in: confirm the reduced scope (algorithm focus; LLC partitioning and RTL moved to future work) is acceptable | Notes recorded |

### 0.4 Reading list

| Paper | Venue | Reader | Used in |
|---|---|---|---|
| Efficient tasks scheduling in multicore systems integrated with hardware accelerators (Xu, Shi, Chen) | J. Supercomputing 2023 | you | 1.8 `xu2023.py` baseline; DAG workloads for 1.9; 2.1 weaknesses the new algorithm targets |
| PCCS: Processor-Centric Contention-aware Slowdown Model for Heterogeneous System-on-Chips (Xu, Belviranli, Shen, Vetter) | MICRO 2021 | you | 1.7 measurements; 1.8 `haxconn_style.py` (its contention model); E1 comparison |
| Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips, "HaX-CoNN" (Dagli, Belviranli) | PPoPP 2024 | you | 1.8 `haxconn_style.py` baseline; 1.9 layer dependencies; 1.10 metrics |
| A Hardware-based HEFT Scheduler Implementation for Dynamic Workloads on Heterogeneous SoCs (Fusco, Hassan, Mack, Akoglu) | VLSI-SoC 2022 | you | 3.1 hardware cost metrics and comparison point (9.144 ns) |
| HTS: A Hardware Task Scheduler for Heterogeneous Systems | arXiv 2019 | you | 3.1 dispatch-unit feasibility (queue and select logic) |
| MoCA: Memory-Centric, Adaptive Execution for Multi-Tenant Deep Neural Networks (Kim et al.) | HPCA 2023 | partner | 3.1 bandwidth counters the algorithm needs; 4.1 throttling (future work) |
| Hardware support for contention tracking in CPU and GPU last-level cache | JSA 2025 | partner | 3.1 cost reference; 4.1 tracking vs. partitioning (future work) |
| Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference | 2025 | partner | 1.7 measurement method; 0.3 parameter justification |
| A fully hardware-managed scheduling architecture for AI accelerators | Springer 2026 | partner | 3.1 dependency tracking cost in hardware |
| Qureshi & Patt, Utility-Based Cache Partitioning (UCP) | MICRO 2006 | partner | 4.1 dynamic partitioning (future work) |

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
- `Kernel`: id, type (`GEMM`, `CPU_ONLY`, …), dims (M, K, N), arrival time, bytes read/written, working-set size (kept for future LLC work), parallelizable flag.
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
- `xu2023.py`: ESHCS, recent same-problem baseline (multicore + accelerators, minimize total execution time, no memory contention). Implemented from the paper; docstring lists every simplification.
- `haxconn_style.py`: HaX-CoNN's contention cost model with a PCCS-style slowdown estimate, run inside the online loop (the full SAT/Z3 solver is out of scope; state this).
- `optimal.py`: exact schedule for small DAGs (brute force, or ILP with `scipy.optimize.milp`) plus lower bounds: total bytes / peak DRAM bandwidth, critical path with standalone times, total work / number of resources. Used to report each policy's optimality gap.
- **Validate:** each policy's decisions on a 5-kernel hand-traced workload match expected assignments; `optimal.py` is never worse than any policy and never below the lower bounds.

### 1.9 Workloads (`sim/workload.py`)
- Real layer shapes (e.g., ResNet-50, BERT-base, MobileNet) in `data/layer_shapes/`; CPU-side kernels (pre/post-processing, memory-bound ops).
- Mixes: accelerator-heavy / balanced / CPU-heavy × low / medium / high contention (arrival rate, sizes).
- Seeded RNG; identical workloads across policies.
- Dependencies (decided after reading Xu 2023 and HaX-CoNN): each kernel has a `deps` list and is released once it has arrived and all of its dependencies have finished; the simulator rejects cycles.
- Two task-graph families: **DNN layer chains** (as in HaX-CoNN) and **DAGs shaped like Xu 2023's** (tree, fork-join, random, FFT, Gaussian elimination). Mix memory-bound and compute-bound kernels so contention matters (when every kernel is memory-bound, makespan = total bytes / bandwidth and no policy can help).
- **Validate:** same seed → identical trace; summary stats per mix; no kernel starts before its dependencies finish.

### 1.10 Metrics (`sim/metrics.py`)
- Makespan (cycles), throughput (kernels or inferences per second, comparable to HaX-CoNN), per-kernel latency (mean, p95, p99), core/accelerator utilization, DRAM bandwidth utilization over time, co-run slowdown, optimality gap (vs. `optimal.py` on small DAGs), scheduler running time per decision.
- **Validate:** metrics on the hand-traced workload match hand values.

### 1.11 Experiment harness (`experiments/`)
- Config-driven runner, parameter sweeps, CSV results in `data/results/`, plotting scripts.
- **E1** characterization: co-run slowdown vs. array size, scratchpad size, DRAM BW; compare the simulator's slowdown against a PCCS-style prediction.
- **E2** scheduling comparison: 6 policies (FIFO, HEFT, Qilin, Xu 2023, HaX-CoNN-style, contention-aware) × workload mixes, with the optimal schedule and lower bounds as reference.
- **E5** design-space sweep: best policy across array sizes (16², 32², 64²) and bandwidth levels.
- **Tier 1 exit:** E1, E2, E5 figures produced; first paper draft outline.

---

## Phase 2 — Tier 2: New contention-aware algorithm (`sim/policies/contention_dp.py`)

Built one piece at a time; each piece is measured on its own (E4 ablation).

### 2.1 Motivation: where ESHCS falls short on this system
| ESHCS step | Weakness | Fix in the new algorithm |
|---|---|---|
| Allocation by speedup `w_sw / w_hw` | Standalone times; the paper assumes no contention, so a memory-bound kernel looks faster on the accelerator than it is | Contention-adjusted costs (2.2) |
| Groups by DAG level, all-or-nothing, then MILP + greedy fill | Coarse; the LUT (space) budget does not fit a one-kernel-at-a-time accelerator | DP allocation (2.3) |
| Static allocation | Cannot react when actual times differ from predictions | Online greedy dispatch with correction (2.4), optional re-plan (2.6) |
| SU/SPR priority | Becomes 0 without deadlines | Critical-path priority + contention penalty (2.5) |

### 2.2 Contention-adjusted cost model
- Time of a kernel on each resource given its co-runners, from `allocate()` (as in `contention_aware.py`) or a PCCS-style slowdown.
- Also the slowdown a kernel **causes** the kernels already running (the current contention-aware policy only counts the slowdown it suffers).
- **Validate:** predictions match the simulator on hand-traced 2- and 3-kernel cases.

### 2.3 DP allocation (offline: which kernels go on the accelerator)
- **Chains:** exact DP. State = (layer i, resource of layer i−1); cost = contention-adjusted time + CPU↔accelerator transfer cost.
- **DAGs / independent kernels:** knapsack-style DP over discretized accelerator time, minimizing `max(CPU load / n_cores, accelerator load)`; replaces ESHCS's MILP + greedy fill.
- **Validate:** DP matches brute force on small instances.

### 2.4 Greedy online dispatch
- Contention-aware earliest finish time, restricted to the DP allocation unless the estimate shows a gain above a threshold for switching resources.
- **Validate:** with an unlimited threshold, behaves exactly like the DP allocation.

### 2.5 Heuristics
- Priority: upward rank (critical path, as in HEFT), since there are no deadlines.
- "Cost I cause others": penalty for the slowdown a placement causes running kernels.
- Pairing: prefer co-running a compute-bound kernel with a memory-bound one over two memory-bound kernels.
- **Validate:** each heuristic changes the decision on a hand-built 3–5 kernel case where it should.

### 2.6 Re-plan (optional)
- When the EWMA correction drifts beyond a threshold, re-run the DP on the remaining kernels.

### 2.7 Experiments
- **E3** schedule quality: new algorithm vs. the 6 baselines and the optimal schedule, across chain and DAG families × low / medium / high contention.
- **E4** ablation: add 2.3, 2.4, 2.5 (each heuristic) and 2.6 one at a time.
- Extend **E5**: does the advantage hold across array sizes (16², 32², 64²), scratchpad sizes and DRAM bandwidths?
- **Tier 2 exit:** E3/E4 figures; paper draft with full evaluation section.

---

## Phase 3 — Tier 3: Hardware cost and energy

### 3.1 Analytical hardware cost (`sim/hw_cost.py`)
- For every scheduler: operations per decision (as a function of queue size, number of resources, DAG size), state kept (tables, counters, bits), estimated decision latency under simple per-operation assumptions.
- Dispatch-unit feasibility: could it be built from HTS-style queue/select logic, HW HEFT_RT finish-time tracking, MoCA-style bandwidth counters, and Springer 2026 dependency tracking? Which parts stay offline (the DP) and which run per decision.
- Report in HW HEFT_RT's form (decision latency, resource cost), with its 9.144 ns as the reference; LLC contention-tracking counters (~0.66% area) as an extra reference point.

### 3.2 Dispatch-latency sensitivity
- Charge 1, 4, 16, 64 cycles per decision in the simulator; find where scheduling overhead cancels the schedule-quality gains.

### 3.3 Energy model (`sim/energy.py`, optional, first to drop)
- Per-op energies (MAC, SRAM access, DRAM access per byte) and active/idle power for cores and accelerator; cite sources (e.g., Horowitz ISSCC 2014, Eyeriss/Accelergy tables).
- **Validate:** energy breakdown for a single kernel matches hand calculation.

### 3.4 Experiments
- **E6** overhead and hardware cost: scheduler running time (as ESHCS reports it) and the hardware cost table for every algorithm.
- **E7** energy across policies (optional): do makespan and energy rankings agree?
- **Tier 3 exit:** hardware cost table and overhead figures in draft.

---

## Phase 4 — Future work (out of scope this semester)

Kept for the paper's Future work section and for continuing the project.

### 4.1 Shared LLC and partitioning (old Tier 2)
- Per-kernel miss-rate curves; capacity-based shared LLC (SoC-level cache, ways); LLC → DRAM coupling (DRAM demand = access rate × miss rate).
- Static and utility-based (UCP) way-partitioning; contention tracking instead of partitioning (JSA 2025); MoCA-style bandwidth throttling.
- Experiments: memory-side only, and scheduling × memory-side (additive, overlapping, or conflicting?).
- Smallest version, only if everything else is done: a static LLC split (25/50/75% to the accelerator) as one more knob in E5.

### 4.2 SystemVerilog dispatch unit (old Tier 4)
- Fixed-point golden model of the chosen algorithm's online part; RTL modules (`kernel_queue.sv`, `history_table.sv`, `bw_monitor.sv`, `dispatch_unit.sv`).
- Verilator co-simulation against the golden model (bit-exact), Yosys synthesis, area/timing vs. HW HEFT_RT; measured latency fed back into the simulator.

---

## Phase 5 — Paper and presentation (continuous)

| When | Paper section |
|---|---|
| After Phase 0 | Introduction, motivation, related work |
| After Tier 1 | Methodology (simulator, systolic model, measurements), E1/E2/E5 |
| After Tier 2 | New algorithm design, E3/E4 |
| After Tier 3 | Hardware cost, overhead, energy, E6/E7 |
| End | Conclusions, limitations, future work (LLC partitioning, RTL), AI-use disclosure, presentation slides |


## Working rules

- One step at a time; each step has a validation test that must pass before moving on.
- All hardware parameters live in `sim/config.py` with cited sources.
- All experiments seeded and reproducible from a config file.
- Commit after each validated step.
