"""
this is our planned algorithm: offline + online greedy dispatch, on the same 2 phase skeleton as eshcs fro the paper

offline which will be once, before anything runs, needs the whole dag, instead of one at a time like xu paper
-plan: which resource each kernel should use, for now a placeholder, but then the ilp replaces it later with the same output
-rank: upward rank, whic means that the longest chain of planned times fro a kernel to the end of the dag.
=> a high rank means a lot of work still waits behind this kernel, so it would go first

online:
-the ready kernels in rank order, each goes on its planned resource if that one is free
-override: if the planned resource is busy and the other one is free, ask the cost model how long it would take there right now
=> start it there only if that beats waiting for the planned one

the heuristics (malleable cores, feeding, delay) come later after off and on phases

"""

from sim.kernels import cpu, accelerator, topological
from sim.estimate import fits_accel, standalone_ns
from sim.policies.base import Policy
from sim.policies.cost_model import what_if, running_lanes, play, Lane, segments
from sim.policies.eshcs import successors, earliest_start, critical_path, latest_finish, successor_rank, priorities
from sim.memory import allocate


#{kernel id: {resource: ns alone}}, only the resources the kernel can run on. every rank and the
#plan read their times from here, so swapping in contention-adjusted times later changes all of them
def cost_table(kernels, sim):
    cost = {}
    for k in kernels:
        cost[k.kernel_id] = {cpu: standalone_ns(k, cpu, sim)}
        if fits_accel(k, sim):
            cost[k.kernel_id][accelerator] = standalone_ns(k, accelerator, sim)
    return cost

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



#placeholder plan: {kernel id: cpu or accelerator} whatever is faster alone with 1 core
def core_plan(kernels, cost):
    plan = {}
    for k in kernels:
        c = cost[k.kernel_id]
        on_accel = accelerator in c and c[accelerator] < c[cpu]
        plan[k.kernel_id] = accelerator if on_accel else cpu 
    return plan


#how many other kernels run next to a kernel on average: tot work / longest chain - itself, capped at the cores. 
#fraction => 0.4 = it shares dram for 40% of its run
def overlap(kernels, plan, cost, n_cores):
    work = sum(cost[k.kernel_id][plan[k.kernel_id]] for k in kernels)
    span = max(upward_rank(kernels, plan, cost).values())
    return min(n_cores, max(0.0, work / span - 1))

#bytes per sec a typical co runner pulls: each kernel's bytes / its time alone avg
def typical_demand(kernels, plan, cost, sim):
    bws = []
    for k in kernels:
        kid = k.kernel_id
        nbytes = sum(s.bytes for s in segments(k, plan[kid], sim))
        bws.append(nbytes / cost[kid][plan[kid]] * 1e9)
    return sum(bws) / len(bws)

#ns on this resource next to others typical corunners, segment by segment like the simulator
def shared_ns(kernel, where, sim, others, demand):
    total = 0.0
    for seg in segments(kernel, where, sim):
        wants = {"me": seg.demand()}
        for i in range(others):
            wants[i] = demand
        bw = allocate(wants, sim.config.memory, sim.degradation)["me"]
        total += seg.duration_ns(bw)
    return total


#same shape as cost table, but each time counts the expected dram sharing
def adjusted_cost_table(kernels, sim):
    alone = cost_table(kernels, sim)
    plan = core_plan(kernels, alone)
    others = overlap(kernels, plan, alone, sim.config.cores.n_cores)
    demand = typical_demand(kernels, plan, alone, sim)
    lo, frac = int(others), others - int(others)
    cost = {}
    for k in kernels:
        kid = k.kernel_id
        cost[kid] = {}
        for where in alone[kid]:
            t = shared_ns(k, where, sim, lo, demand)
            if frac > 0:
                t = (1 - frac) * t + frac * shared_ns(k, where, sim, lo + 1, demand)
            cost[kid][where] = t
    return cost

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

#ns from now till the first kernel on this resource finishes (= it gets free), counting the ones
#already picked in this pick() as running
def free_in(kind, sim, starting):
    lanes = running_lanes(sim)
    on = {kid: sim.by_id[kid].resource for kid in lanes}
    for k, where, n in starting:
        lanes[k.kernel_id] = Lane(segments(k, where, sim, n or 1))
        on[k.kernel_id] = where
    finish = play(lanes, sim.config.memory, sim.degradation)
    return min((t for kid, t in finish.items() if on[kid] == kind), default=0.0)


class ContentionILP(Policy):
    def __init__(self, kernels, override=True, priority="upward", adjusted=True):
        if priority not in PRIORITY:
            raise ValueError(f"Unknown priority: {priority}, pick one of {list(PRIORITY)}")
        self.kernels = topological(kernels)
        self.override = override
        self.priority = priority
        self.cost = None #resource in kernel id
        self.plan = None #{kernel id: cpu or accelerator}, made on the 1st on_queued
        self.rank = None #{kernel id: upward rank in ns}
        self.n_desc = None #how many kernels come after it, break ties in rank
        self.ready = [] #released, not started yet
        self.adjusted = adjusted

    #offline phase, runs once. needs sim for the hardware, so it waits for the 1st arrival

    def on_queued(self, kernel, sim):
        if self.plan is None:
            self.prepare(sim)
        self.ready.append(kernel)

    #where to start this kernel right now, or None = let it wait
    def choose(self, kernel, free, sim, picks):
        planned = self.plan[kernel.kernel_id]
        if free[planned]:
            return planned
        other = cpu if planned == accelerator else accelerator
        if not self.override or not free[other]:
            return None
        if other == accelerator and not fits_accel(kernel, sim):
            return None
        there_now = what_if(kernel, other, sim, starting=picks).time
        wait_here = free_in(planned, sim, picks) + self.cost[kernel.kernel_id][planned]
        return other if there_now < wait_here else None

    #sort key: highest rank first, then the one with more kernels after it, then the lowest id
    def order(self, kernel):
        kid = kernel.kernel_id
        return (-self.rank[kid], -self.n_desc[kid], kid)

    def pick(self, sim):
        picks = []
        free = {cpu: len(sim.cpu.free_cores()) > 0, accelerator: sim.accelerator.free}
        free_cores = len(sim.cpu.free_cores())

        for kernel in sorted(self.ready, key=self.order):
            if not free[cpu] and not free[accelerator]:
                break
            kind = self.choose(kernel, free, sim, picks)
            if kind is None:
                continue
            if kind == accelerator:
                picks.append((kernel, accelerator, None))
                free[accelerator] = False
            else:
                picks.append((kernel, cpu, 1))
                free_cores -= 1
                free[cpu] = free_cores > 0
            self.ready.remove(kernel)

        return picks

    def prepare(self, sim):
        table = adjusted_cost_table if self.adjusted else cost_table
        self.cost = table(self.kernels, sim)
        self.plan = core_plan(self.kernels, self.cost)
        self.rank = PRIORITY[self.priority](self.kernels, self.plan, self.cost, sim)
        self.n_desc = {kid: len(d) for kid, d in descendants(self.kernels).items()}


