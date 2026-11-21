"""
priorities for the new algorithm (step 1): one score per kernel, higher goes first
- upward, lower_bound, eshcs, picked with ContentionILP(priority=...)
- descendants: the tie-break, and part of lower_bound
"""

from sim.policies.cost_model import segments
from sim.policies.eshcs import successors, earliest_start, critical_path, latest_finish, successor_rank, priorities


#{kernel id: set of every kernel after it}, each one counted once. kernels = topological order
def descendants(kernels):
    suc = successors(kernels)
    des = {}
    for k in reversed(kernels): #child come first
        kid = k.kernel_id
        des[kid] = set()
        for c in suc[kid]:
            des[kid] |= {c} | des[c]
    return des


#upward rank: own planned time + the biggest rank of any child
#kernels = topological order
def upward_rank(kernels, plan, cost, sim=None):
    child = {k.kernel_id: [] for k in kernels}
    for k in kernels:
        for d in k.deps:
            child[d].append(k.kernel_id)
    rank = {}
    for k in reversed(kernels): #child come first
        kid = k.kernel_id
        rank[kid] = cost[kid][plan[kid]] + max((rank[c] for c in child[kid]), default=0.0)
    return rank

"""
Kernels only store their parents (deps). The first loop flips that into children.
reversed(kernels) only works if kernels is in topological order, with every kernel after its dependencies. The class guarantees that by calling topological() in __init__.
In reverse order every child comes before its parent, so rank[c] is always ready when the parent needs it.
default=0.0 covers exit kernels, which have no children: their rank is just their own time.

"""


#lower bound rank: the least time to finish this kernel and everything after it, the biggest of
#- the longest chain (upward rank)
#- all their bytes through dram at peak bandwidth
#- all their work spread over every core + the accelerator

def lower_bound_rank(kernels, plan, cost, sim):
    up = upward_rank(kernels, plan, cost)
    desc = descendants(kernels)
    peak = sim.config.memory.dram_bw_bytes_per_s
    lanes = sim.config.cores.n_cores + 1
    nbytes = {k.kernel_id: sum(s.bytes for s in segments(k, plan[k.kernel_id], sim)) for k in kernels}
    rank = {}
    for k in kernels:
        kid = k.kernel_id
        group = desc[kid] | {kid}
        by_bytes = sum(nbytes[d] for d in group) / peak * 1e9
        by_work = sum(cost[d][plan[d]] for d in group) / lanes
        rank[kid] = max(up[kid], by_bytes, by_work)
    return rank


#eshcs rank: the partner's priority (eqs 5 to 11 in eshcs.py), with et = time on the planned resource.
#static: computed once at now = 0
def eshcs_rank(kernels, plan, cost, sim=None, now=0.0):
    et = {k.kernel_id: cost[k.kernel_id][plan[k.kernel_id]] for k in kernels}
    succ = successors(kernels)
    est = earliest_start(kernels, et)
    lft = latest_finish(kernels, et, succ, critical_path(est, et))
    return priorities(kernels, et, est, lft, successor_rank(succ), succ, now)

PRIORITY = {"upward": upward_rank, "lower_bound": lower_bound_rank, "eshcs": eshcs_rank}

"""
lower_bound_rank:

group is the kernel plus everything after it: all the work that can’t finish until this kernel starts.
by_bytes: all of group’s DRAM traffic at peak bandwidth, in ns (* 1e9 converts seconds to ns). The bytes come from the same segments the simulator runs on the planned resource, the same way your partner’s lower_bounds counts them.
by_work: group’s total time spread across every core plus the accelerator.
The rank is the largest of the three. Its first term is the upward rank, so it’s never below the upward rank (a test checks this).

eshcs_rank is the wrapper we agreed on. now=0.0 makes it the static version, and the extra now argument is how a dynamic version could pass the current time later.

"""
