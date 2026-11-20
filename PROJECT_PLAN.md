# Contention-Aware Scheduling for Shared-Memory CPU–Accelerator SoCs

**CPE 5300 Course Project — Implementation Plan (Tiers 1–3)**

## Research question

In a shared-memory SoC where CPU cores and a systolic-array accelerator compete for DRAM bandwidth, can a scheduler that models this contention produce better schedules than existing heterogeneous schedulers that ignore it, and what would it cost to run as an **on-chip dispatch unit**? We compare FIFO, HEFT, Qilin and Xu 2023 (ESHCS) against a **new contention-aware algorithm** (offline: contention-aware ILP allocation + a priority per kernel; online: a rollout search over a greedy dispatcher, tuned with contention heuristics; on the same two-phase skeleton as ESHCS so every gain is attributable), measure how close each one gets to the optimal schedule, and estimate each algorithm's hardware cost. The answer is studied across accelerator sizes and memory bandwidths.

Project type (syllabus): *compare existing ideas that have not been directly compared* + *initial study of a new idea* (the contention-aware algorithm).

Goal metric: **schedule quality** (makespan and latency under contention) first; scheduling overhead (time per decision) second.

## Scope change (2026-11-09)

With about 5 weeks left, the scope is cut to the simulator, the new algorithm, and an analytical hardware cost:
- **Shared LLC and partitioning** (old Tier 2): moved to Future work (Phase 4).
- **SystemVerilog dispatch unit** (old Tier 4): moved to Future work (Phase 4), replaced by the analytical hardware cost in Tier 3.
- **Workloads and baselines (2026-11-12):** workloads are reduced to the five DAG families from Xu 2023 (tree, fork-join, random, FFT, Gaussian elimination), so the comparison against ESHCS is on its own benchmarks. DNN layer chains and the HaX-CoNN-style baseline (built for DNN chains) are dropped.
- The architecture side of the project comes from the contention model (shared DRAM bandwidth, memory-bound vs. compute-bound systolic array), the design-space sweep (array size, scratchpad, bandwidth), and the hardware cost of each scheduler.

## Tier overview

| Tier | Scope | Experiments enabled |
|---|---|---|
| **1 — Core** | Event engine, cores, phase-based systolic accelerator, DRAM bandwidth contention (grounded in C measurements), task dependencies, Xu 2023 DAG workloads, baseline policies (FIFO, HEFT, Qilin, Xu 2023) + contention-aware policy, optimal schedule and lower bounds, metrics, sweeps | E1 characterization, E2 scheduling comparison, E5 design-space sweep |
| **2 — New algorithm** | Cost model (time + harm) + whole-DAG ILP allocation (dependencies, load, DRAM traffic) + pluggable priority (upward rank, ESHCS, lower-bound rank) + greedy dispatch + rollout search on top of the greedy + contention heuristics behind a bandwidth trigger ("cost I cause others", compute/memory pairing), built and measured one piece at a time | E3 quality vs. baselines and optimal, E4 ablation, E5 extended |
| **3 — Hardware cost + energy** | Analytical hardware cost of every scheduler (operations per decision, state, decision latency, dispatch-unit feasibility) vs. HW HEFT_RT; dispatch-latency sensitivity; energy model (optional) | E6 overhead and hardware cost, E7 energy (optional) |

Each tier ends with a complete, writable paper. Never start a tier until the previous tier's validation passes.

## Schedule (5 weeks)

| Week | Work |
|---|---|
| 1 (Nov 9) | Dependencies in the simulator; 1.9 DAG workloads; 1.10 metrics; start `eshcs.py` |
| 2 (Nov 16) | `eshcs.py`, `optimal.py` + lower bounds; 1.11 harness; first E1/E2 runs; 1.7 measurements in parallel |
| 3 (Nov 23, short week) | Tier 2 (cost model and greedy skeleton done Nov 18): (1) pluggable priority (upward, lower-bound, ESHCS) by Nov 22; (2) ILP allocation + contention-adjusted costs; experiment harness; first E3 results |
| 4 (Nov 30) | Tier 2: (3) rollout search `rollout_k` = 0, 1, 2, 3, full; (4) heuristics (malleable cores, feeding, delay, harm/pairing + trigger); (5) re-plan + noise knob if time allows; E4 ablation, E5 sweep; Tier 3 hardware cost table |
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
    eshcs.py                 # ESHCS (Xu 2023), recent same-problem baseline (1.8)
    contention_aware.py      # software form (Tier 1)
    optimal.py               # exact schedule + lower bounds on small DAGs (1.8)
    cost_model.py            # what-if replay: a candidate's time and the harm it causes (2.2)
    contention_ilp.py        # new algorithm (Tier 2)
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
| Efficient tasks scheduling in multicore systems integrated with hardware accelerators (Xu, Shi, Chen) | J. Supercomputing 2023 | you | 1.8 `eshcs.py` baseline; DAG workloads for 1.9; 2.1 weaknesses the new algorithm targets |
| PCCS: Processor-Centric Contention-aware Slowdown Model for Heterogeneous System-on-Chips (Xu, Belviranli, Shen, Vetter) | MICRO 2021 | you | 1.7 measurements; 2.2 PCCS-style slowdown option in the cost model; E1 comparison |
| A Hardware-based HEFT Scheduler Implementation for Dynamic Workloads on Heterogeneous SoCs (Fusco, Hassan, Mack, Akoglu) | VLSI-SoC 2022 | you | 3.1 hardware cost metrics and comparison point (9.144 ns) |
| HTS: A Hardware Task Scheduler for Heterogeneous Systems | arXiv 2019 | you | 3.1 dispatch-unit feasibility (queue and select logic) |
| MoCA: Memory-Centric, Adaptive Execution for Multi-Tenant Deep Neural Networks (Kim et al.) | HPCA 2023 | partner | 3.1 bandwidth counters the algorithm needs; 4.1 throttling (future work) |
| Hardware support for contention tracking in CPU and GPU last-level cache | JSA 2025 | partner | 3.1 cost reference; 4.1 tracking vs. partitioning (future work) |
| Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference | 2025 | partner | 1.7 measurement method; 0.3 parameter justification |
| A fully hardware-managed scheduling architecture for AI accelerators | Springer 2026 | partner | 3.1 dependency tracking cost in hardware |
| Qureshi & Patt, Utility-Based Cache Partitioning (UCP) | MICRO 2006 | partner | 4.1 dynamic partitioning (future work) |

Reading order for you: Xu 2023 and PCCS (needed before 1.8 baselines and 2.2), then HW HEFT and HTS (needed before Tier 3).

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
- `eshcs.py`: ESHCS, recent same-problem baseline (multicore + accelerators, minimize total execution time, no memory contention). Implemented from the paper; docstring lists every simplification.
- `optimal.py`: exact schedule for small DAGs plus lower bounds: total bytes / peak DRAM bandwidth, critical path with standalone times, total work / number of resources. Used to report each policy's optimality gap.
  - Brute force (done) reaches ~8 kernels and only tries list schedules. **Proposed (2026-11-18):** replace it with A* (or depth-first branch and bound: same bounds, less memory) over partial schedules, using `lower_bound` of the remaining kernels as the admissible heuristic and "wait until the next finish" as a move, so it reaches the 14-20 kernel Xu DAGs and finds schedules that idle on purpose. Evaluation only, not part of the new algorithm.
- **Validate:** each policy's decisions on a 5-kernel hand-traced workload match expected assignments; `optimal.py` is never worse than any policy and never below the lower bounds.

### 1.9 Workloads (`sim/workload.py`)
- Each DAG node is a random kernel: a GEMM (compute-bound, CPU or accelerator) or an elementwise op (memory-bound, CPU only).
- Contention knob `gemm_frac`: share of GEMMs; lower = more memory-bound kernels competing for DRAM (low / medium / high contention).
- Seeded RNG; identical workloads across policies.
- Dependencies (decided after reading Xu 2023): each kernel has a `deps` list and is released once it has arrived and all of its dependencies have finished; the simulator rejects cycles.
- The five DAG families from Xu 2023: tree (in and out), fork-join, random layered, FFT, Gaussian elimination; every generator takes a size, `seed`, `gemm_frac` and `start_id`. Mix memory-bound and compute-bound kernels so contention matters (when every kernel is memory-bound, makespan = total bytes / bandwidth and no policy can help).
- **Validate:** same seed → identical trace; summary stats per mix; no kernel starts before its dependencies finish.

### 1.10 Metrics (`sim/metrics.py`)
- Makespan (cycles), throughput (kernels per second), per-kernel latency (mean, p95, p99), core/accelerator utilization, DRAM bandwidth utilization over time, co-run slowdown, optimality gap (vs. `optimal.py` on small DAGs), scheduler running time per decision.
- **Validate:** metrics on the hand-traced workload match hand values.

### 1.11 Experiment harness (`experiments/`)
- Config-driven runner, parameter sweeps, CSV results in `data/results/`, plotting scripts.
- **E1** characterization: co-run slowdown vs. array size, scratchpad size, DRAM BW; compare the simulator's slowdown against a PCCS-style prediction.
- **E2** scheduling comparison: 5 policies (FIFO, HEFT, Qilin, Xu 2023, contention-aware) × DAG families × contention levels, with the optimal schedule and lower bounds as reference.
- **E5** design-space sweep: best policy across array sizes (16², 32², 64²) and bandwidth levels.
- **Tier 1 exit:** E1, E2, E5 figures produced; first paper draft outline.

---

## Phase 2 — Tier 2: New contention-aware algorithm (`sim/policies/contention_ilp.py`)

Same two-phase skeleton as ESHCS (offline allocation + online scheduling), so the comparison is controlled: every difference from ESHCS models contention or fixes a weakness listed in 2.1, and the E4 ablation adds them one at a time starting from ESHCS. Built one piece at a time; each piece is measured on its own.

**Decision (2026-11-14):** ILP instead of DP for the allocation. DP and ILP give the same answer to the same knapsack, so switching solvers alone is not a contribution; the ILP is chosen because it can also model dependencies and DRAM traffic (2.3), which a knapsack DP cannot do easily.

**Decision (2026-11-18): the full algorithm**

```
OFFLINE (once per DAG)
  ILP plan     each kernel -> cpu or accelerator (2.3)
  priority     one score per kernel, pluggable (2.3b): upward | eshcs | lower_bound
               tie-break: descendant count
ONLINE (every pick)
  greedy       ready kernels by priority -> planned resource, cost-model override (2.4)
  rollout      top-k candidates (+ "wait"), each finished with the greedy, keep the best (2.4b)
               k = 0 (plain greedy), 1, 2, 3, full (every ready candidate)
  heuristics   flags, tuned later (2.5): malleable cores, feeding, delay, harm/pairing, trigger
  re-plan      optional (2.6): actual times drift past a threshold -> correct the costs,
               re-solve the ILP on the kernels not started yet, recompute the priority
EVALUATION (not part of the algorithm)
  A* / DFBnB optimum on small DAGs, lower bounds on large ones (1.8)
```

**Build order (2026-11-18):** greedy skeleton (done) → (1) pluggable priority: upward, lower-bound, ESHCS (2.3b) → (2) ILP plan + contention-adjusted costs (2.3) → (3) rollout, `rollout_k` = 0, 1, 2, 3, full (2.4b) → (4) heuristics (2.5) → (5) optional re-plan (2.6), only if the ILP lands on time; otherwise it stays future work (4.3). All offline parts are finished before the rollout, so it is tested on the final plan and ranks.

Rejected for the online part: full A* (exponential per decision, no dispatch-unit story, breaks the step-by-step comparison with ESHCS) and MCTS (the simulator is deterministic, randomness adds nothing). Optional extras if time allows: batch matching (assign all ready kernels at once), PEFT and sufferage as extra baselines, a DRAM price from the ILP's LP dual as the harm weight.

### 2.1 Motivation: where ESHCS falls short on this system
| ESHCS step | Weakness | Fix in the new algorithm |
|---|---|---|
| Allocation by speedup `w_sw / w_hw` | Standalone times; the paper assumes no contention, so a memory-bound kernel looks faster on the accelerator than it is | Contention-adjusted costs (2.2) |
| Groups by DAG level, all-or-nothing, then MILP + greedy fill | Coarse, ignores the critical path; the LUT (space) budget does not fit a one-kernel-at-a-time accelerator | Whole-DAG ILP with dependency, load and DRAM constraints (2.3) |
| Static allocation | Cannot react when actual times differ from predictions | Online greedy dispatch with override (2.4), optional re-plan (2.6) |
| SU/SPR priority | Without deadlines it reduces to critical-path urgency; blind to DRAM traffic | Pluggable priority incl. a contention-aware lower-bound rank (2.3b) |
| One-step dispatch | Each decision only looks at now | Rollout search over the greedy (2.4b) |

### 2.1b How the new algorithm differs from ESHCS
In one sentence: ESHCS treats runtimes as fixed and plans by speedup; the new algorithm treats runtimes as depending on what else is running and plans and dispatches for makespan under shared bandwidth (the slowdown a kernel suffers, the slowdown it causes, and when not to start at all).

**Offline: which GEMMs go on the accelerator**

| | ESHCS | New algorithm | Contention-specific? |
|---|---|---|---|
| Time estimates | Standalone | Contention-adjusted (2.2) | yes |
| Scope | One DAG level at a time, biggest speedup first, all-or-nothing | Whole DAG at once | no |
| Solver | Whole groups, ILP on the first group that does not fit, greedy fill | One ILP (2.3) | no |
| Objective | Speedup within a fixed budget (`hw_share`) | Estimated makespan `T` | no |
| Dependencies | Only through level order | Critical-path constraints on every edge | no |
| Memory traffic | Ignored | DRAM constraint `Σ bytes / peak_bw ≤ T` | yes |
| Budget knob | `hw_share` (tuned) | None; the load constraints decide | no |

**Online: what starts now**

| | ESHCS | New algorithm | Contention-specific? |
|---|---|---|---|
| Priority | SU/SPR (critical-path urgency without deadlines) | Upward rank, or lower-bound rank with downstream DRAM bytes (2.3b) | partly |
| Lookahead | None | Rollout: each candidate finished with the greedy in simulation (2.4b) | partly |
| Placement | Strictly the plan | Plan, with override when the cost model shows a gain (2.4) | partly |
| Runtime estimate | Fixed standalone | What-if replay with the running kernels (2.2) | yes |
| Effect on others | Not considered | Harm (2.2, 2.5) | yes |
| Cores per CPU kernel | 1 | Malleable cores (2.5) | no |
| Accelerator idle time | Not considered | Accelerator feeding (2.5) | no |
| Starting work | Always, if a resource is free | Deliberate delay (2.5) | yes |
| Decision cost | Always cheap | Cheap by default, full cost model only under the contention trigger (2.5) | yes |

**Kept the same on purpose:** the two-phase skeleton (ILP offline, greedy online), GEMM-only one-at-a-time accelerator, the ESHCS DAG families, makespan as the objective.

### 2.2 Contention-adjusted cost model (`sim/policies/cost_model.py`)
- What-if replay of the running jobs with the simulator's own rate model (segments + `allocate()`): for a candidate (kernel, resource) it returns **time** (how long it takes with its co-runners) and **harm** (extra ns it causes the running kernels, the "cost I cause others" that `contention_aware.py` ignores).
- Exact when nothing else starts in the meantime.
- Harm is also returned per running kernel (`{kernel_id: extra_ns}`), so the delay heuristic (2.5) can weight it by criticality.
- **Validate:** time and harm match the simulator on hand-traced 2- and 3-kernel cases (CPU + accelerator, compute- + memory-bound).
- Cheap approximation for hardware (Tier 3): harm ≈ overlap time × the bandwidth fraction the candidate takes; compare against the replay.

### 2.3 ILP allocation (offline: which GEMMs go on the accelerator)
- Variables: `x_v ∈ {0,1}` (GEMM v on the accelerator), `f_v ≥ 0` (estimated finish), `T` (makespan estimate). `cost_v = w_sw_v (1 - x_v) + w_hw_v x_v`.
- Minimize `T` subject to:
  - dependencies: `f_v ≥ f_u + cost_v` for every edge `u -> v`, `f_v ≥ cost_v`, `T ≥ f_v` (critical path)
  - accelerator load: `T ≥ Σ w_hw_v x_v` (one kernel at a time)
  - CPU load: `T ≥ Σ w_sw_v (1 - x_v) / n_cores`
  - DRAM traffic: `T ≥ Σ bytes_v(x_v) / peak_bw` (a GEMM moves different bytes on the accelerator than on the CPU)
- Costs `w_sw`, `w_hw` are contention-adjusted (memory-bound CPU kernels inflated by the expected co-run slowdown from 2.2), not standalone.
- The `f_v` constraints ignore resource conflicts, so `T` is an estimate for planning; the greedy dispatch (2.4) handles the details. With `x` relaxed to [0, 1] the same model gives a lower bound for `optimal.py`.
- Solved with `scipy.optimize.milp`; runs once per DAG (offline). Its solve time is reported as the one-time planning cost, separately from makespan.
- **Validate:** on small DAGs, the ILP placement replayed in the simulator is within a few % of the brute-force optimum; each constraint changes the placement on a hand-built case where it should (deep chain → critical path, many memory-bound kernels → DRAM).

### 2.3b Priority (offline, pluggable)
- One score per kernel, computed once; `ContentionILP(kernels, priority=...)`. Ties broken by descendant count (more kernels unlocked first).
- Every rank function takes a `cost` table `{kernel_id: {cpu: ns, accelerator: ns}}` instead of calling `standalone_ns` itself. `prepare()` builds it once: standalone times first, the contention-adjusted times from 2.3 once they exist. So all three ranks are built now and become contention-aware without changes when 2.3 lands.
- `upward` (default): own planned time + the biggest child rank (critical path, as in HEFT).
- `eshcs`: the partner's eqs 5-11 from `eshcs.py` (`successors` → `earliest_start` → `critical_path` as the deadline → `latest_finish` → `successor_rank` → `priorities`), with `et` = the planned resource's time from `cost`. Same dispatch, only the priority changes, so the ablation isolates the priority. Urgency depends on `now`: first version static (computed once at `now = 0`, equal to urgency at each kernel's earliest start); a `dynamic` option recomputes it in `pick()` for the ready kernels, as the paper does, so the ESHCS comparison is fair.
- `lower_bound` (new, contention-aware): `max(upward rank, downstream bytes / peak_bw, downstream work / (n_cores + 1))`, a lower bound on finishing everything after the kernel (the per-kernel form of `optimal.lower_bounds`). Downstream = every descendant counted once. Built with standalone costs first; uses the adjusted costs automatically after 2.3.
- Compared without rollout first; the best one is used inside the rollout (it chooses the top-k candidates and drives the greedy that finishes every branch).
- **Validate:** hand-built chains for `upward`; a short chain with heavy downstream traffic outranks a longer compute-only chain under `lower_bound` but not under `upward`.

### 2.4 Greedy online dispatch
- **Done (2026-11-18, `sim/policies/p1_ilp.py`):** the greedy skeleton with a placeholder plan (`core_plan`: a GEMM goes on the accelerator if it is faster there alone), upward rank, and the cost-model override. The greedy only *reads* the plan, so the ILP plan (2.3) is swapped in afterwards without changing the online code.
- Ready kernels by priority (upward rank); each goes where the plan placed it, unless that resource is busy, another is free, and the cost model shows a gain above a threshold (override).
- Each heuristic in 2.5 is a constructor flag (e.g. `ContentionILP(delay=False, malleable=False, feed=False)`), so the E4 ablation is the same policy with flags switched on one at a time.
- **Validate:** with an infinite threshold it follows the plan exactly; with every flag off it matches the plain greedy dispatch.

### 2.4b Rollout search (online, on top of the greedy)
- At a pick, the candidates are the top-k ready kernels by priority, each on its planned resource and on the other one if it is free, plus "wait until the next event". For each candidate: copy the simulation state, apply the move, finish the whole DAG with the greedy (2.4), record the makespan. Start the candidate with the best makespan; ties go to the greedy's choice.
- With the greedy as the base policy the rollout is never worse than the greedy (cost-improvement property, Bertsekas). "Wait" as a move makes it find deliberate delays on its own, a principled version of heuristic 3.
- State copy: `copy.deepcopy(sim)` first; fallback: record every `pick()` result and replay the prefix from t = 0 (the simulator is deterministic).
- `rollout_k` = 0 (off, plain greedy), 1, 2, 3, or `"full"` (every ready kernel on every free resource, plus "wait"; still a one-step rollout, not a full tree search, which is the A* optimum). Optionally run it only when the contention trigger fires, to bound the cost.
- Cost: about decisions × candidates × one full simulation. Software only, not a dispatch-unit design; reported as a quality vs. `sched_ns` curve over k.
- **Validate:** a rollout with no changes reproduces the greedy's makespan exactly; never worse than the greedy on all five families and several seeds; finds the delay win on the hand-built case from 2.5 heuristic 3.

### 2.5 Heuristics (behind a contention trigger, tuned after the rollout)
- **Contention trigger:** the contention heuristics run only when `running demand + candidate demand > peak bandwidth`; otherwise the cheap path (plan + priority, about ESHCS's cost) decides. Keeps the average decision cost low for Tier 3.
- "Cost I cause others": pick the candidate minimizing `time + λ · harm` from 2.2.
- Pairing: prefer co-running a compute-bound kernel with a memory-bound one over two memory-bound kernels.

**Chosen improvements (2026-11-14), built in this order, each on its own branch with its own tests:**

1. **Malleable cores** (`malleable`, first, easiest). When more cores are free than kernels waiting for the CPU, give a splittable kernel extra idle cores. Extra cores only speed up compute, so a memory-bound kernel gains nothing: ask the cost model `what_if(k, cpu, sim, num_cores=n)` for n = 1 .. free cores and take the smallest n within a few % of the best time. Variant for the ablation: keep cores free for children of kernels about to finish.
   - **Validate:** a lone compute-bound kernel takes 2 cores and finishes faster; a lone memory-bound kernel stays on 1 core; flag off = 1-core behavior.
2. **Accelerator feeding** (`feed`). Priority bonus for ready CPU kernels whose children are GEMMs planned for the accelerator, larger when the accelerator is idle or about to be, so it does not sit waiting for inputs.
   - **Validate:** of two equal-rank CPU kernels, the one that unlocks an accelerator GEMM starts first; the GEMM starts earlier and `accel_util` goes up.
3. **Deliberate delay** (`delay`, last, most distinctive). Allow "start nothing now". Starting a candidate costs its harm, but only harm to critical-path kernels lengthens the makespan, so weight each harmed kernel's extra ns by its criticality (upward rank); waiting costs at most the time until the next event, and nothing for a candidate with slack. Delay when weighted harm > the cost of waiting. Without the criticality weighting it delays too often and gets worse.
   - **Validate (hand-built):** k0 memory-bound at the head of a long chain (k0 → k2 → k3, k2 compute-bound), k1 memory-bound, short, no children. Without delay k1 slows k0 and the whole chain; with delay k1 waits for k0 and then co-runs with compute-bound k2, and the makespan is shorter. Flag off = plain greedy.

### 2.6 Re-plan (optional, decided 2026-11-20)
- **Why:** the plan is built on estimates (contention-adjusted costs are averages over expected co-runners; the ILP's `T` ignores who waits for which resource). On real hardware predictions are never exact, so a plan that cannot change goes stale.
- **Drift check:** when a kernel finishes, compare its actual time with `cost[kid][resource]`; keep a running error (EWMA, as in `contention_aware.py`). Re-plan when the error passes a threshold (e.g. 10-20%), at most once per finished kernel.
- **Re-plan:**
  1. Correct the cost table with the observed error, so the new plan does not repeat the old mistake.
  2. Re-solve the ILP (2.3) on the kernels that have not started; finished and running kernels are fixed, and their (predicted) finish times become earliest-start constraints for their children.
  3. Recompute the priority (2.3b) on the new plan.
- The online part (greedy, rollout, heuristics) is unchanged; it reads the new `self.plan`, `self.cost` and `self.rank`. Constructor flag `replan=False` (off by default) plus the threshold.
- **Drift in the simulator:** the simulator runs the same rate model the cost model replays, so drift only comes from the plan's estimates. To show the value of re-planning, add a noise knob: actual segment times = predicted × a random factor (±10, 25, 50%, seeded). The policy only sees the predictions.
- **Cost:** one ILP solve per trigger (small for 14-20 kernel DAGs); report the number of re-plans and total solve time next to the makespan. Stays on the software / host side, like the ILP (Tier 3).
- **Validate:** threshold = infinity reproduces the run without re-planning exactly; with no noise it rarely or never triggers; on a hand-built case where one kernel takes 2× its estimate, the re-plan moves a later kernel to the other resource and the makespan is shorter.

### 2.7 Experiments
- **E3** schedule quality: new algorithm vs. the 5 baselines and the optimal schedule, across the five DAG families × low / medium / high contention.
- **E4** ablation, starting from ESHCS: + whole-DAG ILP with dependencies → + DRAM constraint and adjusted costs → + priority (ESHCS → upward → lower-bound) → + harm → + pairing → + malleable cores → + accelerator feeding → + deliberate delay → + trigger (same quality, lower average decision cost) → + re-plan → + rollout (k = 1, 2, 3, full; software only). Compare rollout's found delays against the deliberate-delay heuristic. Each step is a flag on the same policy. Report the gain from the non-contention steps (whole-DAG ILP, critical-path priority, malleable cores, feeding) separately from the contention-specific ones (DRAM constraint, adjusted costs, harm, delay, trigger), so the paper states how much of the improvement comes from modeling contention.
- Fairness and robustness: sweep ESHCS's `hw_share` and report its best; sweep `λ`, the trigger threshold and the override threshold; add noise to the cost model's predictions to see how much accuracy the gains need. With re-plan (2.6): makespan with and without re-planning as the noise grows (±10, 25, 50%).
- Headroom check first: before tuning heuristics, measure how far ESHCS, HEFT and `contention_aware.py` are from optimal (A*/DFBnB) across bandwidths. First data point (2026-11-18, greedy skeleton with placeholder plan): 0-17% above the lower bound at 10 GB/s, at most 3% at 2 GB/s on the default DAG sizes; if the gap is small at normal bandwidth, the main result is *when* contention-aware scheduling pays off.
- Extend **E5**: does the advantage hold across array sizes (16², 32², 64²), scratchpad sizes and DRAM bandwidths?
- **Tier 2 exit:** E3/E4 figures; paper draft with full evaluation section.

---

## Phase 3 — Tier 3: Hardware cost and energy

### 3.1 Analytical hardware cost (`sim/hw_cost.py`)
- For every scheduler: operations per decision (as a function of queue size, number of resources, DAG size), state kept (tables, counters, bits), estimated decision latency under simple per-operation assumptions.
- Dispatch-unit feasibility: could it be built from HTS-style queue/select logic, HW HEFT_RT finish-time tracking, MoCA-style bandwidth counters, and Springer 2026 dependency tracking? Which parts stay offline (the ILP, the priority) and which run per decision; the cost of the contention trigger vs. the full cost model. The rollout is software only: report its cost per decision next to the greedy's, as the price of the extra quality.
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

### 4.3 Re-planning (if not built in 2.6)
- Re-solve the ILP mid-run on the unstarted kernels when actual times drift from the plan; evaluated under noisy predictions. Design in 2.6.

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
