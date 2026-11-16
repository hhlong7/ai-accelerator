"""
eshcs.py is the ESHCS baseline (1.8) from Xu, Shi, Chen, "Efficient tasks scheduling in multicore
systems integrated with hardware accelerators", J. Supercomputing 2023. its the most recent
scheduler for the same problem (cpu cores + accelerator, minimize makespan), but it ignores memory
contention. it has 2 phases:

1. allocation (offline, once for the whole DAG): which kernels go on the accelerator
   - split the dag into groups by level, the kernels in a group dont depend on each other
   - speedup of a kernel = w_sw / w_hw (time on a core / time on the accelerator)
   - take the group with the biggest total speedup: if all of it fits in the hardware budget,
     it all goes on the accelerator, then the next biggest group, and so on
   - the first group that doesnt fit: pick the best part of it with an ilp (eq 4 in the paper)
   - whatever budget is left: greedy fill, the leftover kernels with the biggest speedup first
2. scheduling (online): ready cpu kernels go to free cores by priority (part B), ready
   accelerator kernels go on the accelerator


changes from the paper to our version:
- the paper's fpga runs multiple tasks in parallel and its budget is LUTs (space). our accelerator
  runs one kernel at a time, so the budget is accelerator TIME: a kernel costs w_hw, and the
  budget is hw_share * (total w_hw of the kernels that could go there). hw_share plays the role
  of the paper's "hardware percentage" (fig 10)
- the paper's hardware is always faster (speedup 2 to 8). here only gemms that are faster on the
  accelerator than on a core can go there, everything else stays on the cpu
- groups are dag levels (level = 1 + the biggest level of its deps), the paper builds them by
  taking successors of the last group and dropping the ones that depend on each other
- the greedy fill skips a kernel that doesnt fit and keeps going down the list
"""


import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds



# groups by DAG level. level 0 = no dependencies
#                      level n = 1 + biggest dep
def levels(kernels):
    level = {}

    for kernel in kernels: 
        level[kernel.kernel_id] = 1 + max((level[d] for d in kernel.deps), default = -1)
    groups = [[] for _ in range(max(level.values()) + 1)]

    for kid, lv in level.items():
        groups[lv].append(kid)

    return groups


#
def knapsack(ids, gain, cost, budget):
    if not ids or budget <= 0:
        return set()
    c = -np.array([gain[i] for i in ids])
    a = np.array([[cost[i] for i in ids]])

    res = milp(c, constraints = LinearConstraint(a, -np.inf, budget),
               integrality = np.ones(len(ids)), bounds = Bounds(0,1))

    return {i for i, x in zip(ids, res.x) if x > 0.5}


# allocation phase. w_hw has kernels that are 
# allowed on the accel. returns a set of kernel ids
# that go on the accel
def allocate_hw(groups, w_sw, w_hw, budget):
    speedup = {i: w_sw[i] / w_hw[i] for i in w_hw}
    pending = [[i for i in g if i in speedup] for g in groups]
    pending = [g for g in pending if g]
    hw, left = set(), budget

    while pending:
        best = max(pending, key = lambda g: sum(speedup[i] for i in g))
        pending.remove(best)
        cost = sum(w_hw[i] for i in best)

        if cost <= left: # if whole group fits
            hw.update(best)
            left -= cost
            continue

        # if doesn't fit
        saved = {i: w_sw[i] - w_hw[i] for i in best}
        chosen = knapsack(best, saved, w_hw, left)
        hw.update(chosen)
        left -= sum(w_hw[i] for i in chosen)
        break

    # then greedy fill with groups that were
    # unchecked
    for i in sorted((i for g in pending for i in g), key = lambda i: -speedup[i]):
        if w_hw[i] <= left:
            hw.add(i)
            left -= w_hw[i]
    return hw



"""

Part B: priorities for the cpu kernels (eqs 5 to 11 in the paper). 
'et' is the execution time of each kernel on 
the resource the allocation gave it
(w_sw or w_hw).

no deadlines for our workloads: 
with no deadline, LFT = infinity, 
urgency = 0 for every kernel and
every priority is 0. so every exit kernel 
gets the same deadline = the critical path 
length (the makespan with unlimited cores).
a kernel on the critical path then has urgency
1 from the start. no communication cost either
(c_ij = 0), the cores and the accelerator 
share memory.

"""


# successors of every kernel
def successors(kernels):
    succ = {k.kernel_id: [] for k in kernels}
    for k in kernels:
        for d in k.deps:
            succ[d].append(k.kernel_id)
    return succ

# eq 7: earliest start time, after release time and every dep finished
def earliest_start(kernels, et):
    est = {}
    for k in kernels: # deps come first
        est[k.kernel_id] = max([k.queue] + [est[d] + et[d] 
                                            for d in k.deps] )
    return est

# the deadline we give every kernel
def critical_path(est, et):
    return max(est[i] + et[i] for i in est)

# eq 8: latest finish time
def latest_finish(kernels, et, succ, deadline):
    lft = {}
    for k in reversed(kernels):
        kid = k.kernel_id
        lft[kid] = min([deadline] + [lft[j] - et[j]
                                     for j in succ[kid]])

    return lft

# eq 5: successor parallel rank
# more succ = more work released when fin
def successor_rank(succ):
    most = max(len(s) for s in succ.values())

    return {kid: (len(s) / most if most else 0.0)
            for kid, s in succ.items()}



# eq 9: scheduling urgency
# 1 = must start now or successor misses deadline
# It is clamped to 1
def urgency(et_i, est_i, lft_i, now):
    window = lft_i - max(est_i, now)
    if window <= et_i:
        return 1.0
    return et_i / window

# eq 11: priority = own urgency + successor rank
def priorities(kernels, et, est, lft, spr, succ, now):
    prio = {}
    for k in reversed(kernels): # successors come first
        kid = k.kernel_id
        su = urgency(et[kid], est[kid], lft[kid], now)
        best_next = max((prio[j] for j in succ[kid]),
                        default = None)
        prio[kid] = su if best_next is None else su + spr[kid] * best_next

    return prio


