"""
plans for the new algorithm (step 2b): which resource each kernel goes on
- core_plan: whatever is faster alone (the placeholder, and the ilp=False ablation)
- ilp_plan: one whole-dag ilp
"""

from sim.kernels import cpu, accelerator
from sim.policies.cost_model import segments
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


#the ilp plan: which resource each kernel goes on, so the estimated makespan T is as small as possible
#all variables sit in one vector: [x_0 .. x_n-1 | f_0 .. f_n-1 | T]
#- x_i = 1 if kernel i goes on the accelerator, 0 = cpu
#- f_i = its estimated finish (ns), T = the estimated makespan
#kernel i takes cpu_i + (acc_i - cpu_i) * x_i, every constraint is written as  row . vars >= low
#relax=True lets x be anything in [0, 1], which gives a lower bound instead of a plan
#placeholder plan: {kernel id: cpu or accelerator} whatever is faster alone with 1 core


def core_plan(kernels, cost):
    plan = {}
    for k in kernels:
        c = cost[k.kernel_id]
        on_accel = accelerator in c and c[accelerator] < c[cpu]
        plan[k.kernel_id] = accelerator if on_accel else cpu 
    return plan


def ilp_plan(kernels, cost, sim, relax=False):
    n = len(kernels)
    X, F, T = 0, n, 2 * n #where each group starts in the vector
    size = 2 * n + 1
    pos = {k.kernel_id: i for i, k in enumerate(kernels)}
    peak = sim.config.memory.dram_bw_bytes_per_s
    n_cores = sim.config.cores.n_cores

    fits = np.zeros(n) #if it can go on the accelerator
    on_cpu, on_acc = np.zeros(n), np.zeros(n) #ns on each resource
    dram_cpu, dram_acc = np.zeros(n), np.zeros(n) #ns its bytes keep dram busy at peak
    for i, k in enumerate(kernels):
        c = cost[k.kernel_id]
        on_cpu[i] = c[cpu]
        dram_cpu[i] = sum(s.bytes for s in segments(k, cpu, sim)) / peak * 1e9
        if accelerator in c:
            fits[i] = 1
            on_acc[i] = c[accelerator]
            dram_acc[i] = sum(s.bytes for s in segments(k, accelerator, sim)) / peak * 1e9
        else: #x_i is held at 0 below, so these copies never count
            on_acc[i], dram_acc[i] = on_cpu[i], dram_cpu[i]
    change = on_acc - on_cpu #how much its time changes when it moves to the accelerator

    rows, low = [], []
    def add(row, at_least):
        rows.append(row)
        low.append(at_least)

    for i, k in enumerate(kernels):
        #it cant finish before its own time: f_i - change_i x_i >= cpu_i
        r = np.zeros(size); r[F + i] = 1; r[X + i] = -change[i]
        add(r, on_cpu[i])
        #or before a dep finishes + its own time: f_i - f_d - change_i x_i >= cpu_i
        for d in k.deps:
            r = np.zeros(size); r[F + i] = 1; r[F + pos[d]] = -1; r[X + i] = -change[i]
            add(r, on_cpu[i])
        #the makespan is after every finish: T - f_i >= 0
        r = np.zeros(size); r[T] = 1; r[F + i] = -1
        add(r, 0.0)

    #the accelerator runs one kernel at a time: T - sum acc_i x_i >= 0
    r = np.zeros(size); r[T] = 1; r[X:F] = -on_acc
    add(r, 0.0)
    #the cpu work spread over the cores: T - sum cpu_i (1 - x_i) / n_cores >= 0
    r = np.zeros(size); r[T] = 1; r[X:F] = on_cpu / n_cores
    add(r, on_cpu.sum() / n_cores)
    #all the bytes through dram at peak: T - sum (dram_acc_i - dram_cpu_i) x_i >= sum dram_cpu_i
    r = np.zeros(size); r[T] = 1; r[X:F] = -(dram_acc - dram_cpu)
    add(r, dram_cpu.sum())

    #minimize T. many plans tie on T (only the critical kernels are pinned), so a tiny weight on
    #the finishes breaks the tie toward every kernel finishing early
    objective = np.zeros(size)
    objective[T] = 1.0
    objective[F:T] = 1e-3 / n

    integrality = np.zeros(size)
    if not relax:
        integrality[X:F] = 1 #x is 0 or 1
    upper = np.full(size, np.inf)
    upper[X:F] = fits #x_i = 0 if it doesnt fit the accelerator

    res = milp(objective, constraints=LinearConstraint(np.array(rows), low, np.inf),
               integrality=integrality, bounds=Bounds(0, upper))
    if not res.success:
        raise RuntimeError(f"ILP failed: {res.message}")
    plan = {k.kernel_id: accelerator if res.x[X + i] > 0.5 else cpu for i, k in enumerate(kernels)}
    return plan, res.x[T]
