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


#placeholder plan: {kernel id: cpu or accelerator} whatever is faster alone with 1 core
def core_plan(kernels, sim):
    plan = {}
    for k in kernels:
        on_accel = fits_accel(k, sim) and standalone_ns(k, accelerator, sim) < standalone_ns(k, cpu, sim)
        plan[k.kernel_id] = accelerator if on_accel else cpu 
    return plan


#upward rank: own planned time + the biggest rank of any child
#kernels = topological order
def upward_rank(kernels, plan, sim):
    child = {k.kernel_id: [] for k in kernels}
    for k in kernels:
        for d in k.deps:
            child[d].append(k.kernel_id)
    rank = {}
    for k in reversed(kernels): #child come first
        kid = k.kernel_id
        rank[kid] = standalone_ns(k, plan[kid], sim) + max((rank[c] for c in child[kid]), default=0.0)
    return rank

"""
Kernels only store their parents (deps). The first loop flips that into children.
reversed(kernels) only works if kernels is in topological order, with every kernel after its dependencies. The class guarantees that by calling topological() in __init__.
In reverse order every child comes before its parent, so rank[c] is always ready when the parent needs it.
default=0.0 covers exit kernels, which have no children: their rank is just their own time.

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
    def __init__(self, kernels, override=True):
        self.kernels = topological(kernels)
        self.override = override
        self.plan = None #{kernel id: cpu or accelerator}, made on the 1st on_queued
        self.rank = None #{kernel id: upward rank in ns}
        self.ready = [] #released, not started yet

    #offline phase, runs once. needs sim for the hardware, so it waits for the 1st arrival
    def prepare(self, sim):
        self.plan = core_plan(self.kernels, sim)
        self.rank = upward_rank(self.kernels, self.plan, sim)

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
        wait_here = free_in(planned, sim, picks) + standalone_ns(kernel, planned, sim)
        return other if there_now < wait_here else None

    def pick(self, sim):
        picks = []
        free = {cpu: len(sim.cpu.free_cores()) > 0, accelerator: sim.accelerator.free}
        free_cores = len(sim.cpu.free_cores())

        for kernel in sorted(self.ready, key=lambda k: -self.rank[k.kernel_id]):
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