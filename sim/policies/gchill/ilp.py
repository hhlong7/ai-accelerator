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

import time
from sim.kernels import cpu, accelerator, topological
from sim.estimate import fits_accel
from sim.policies.base import Policy
from sim.policies.cost_model import what_if, running_lanes, play, Lane, segments
from sim.policies.gchill.costs import cost_table, adjusted_cost_table
from sim.policies.gchill.plan import core_plan, ilp_plan
from sim.policies.gchill.priority import descendants, PRIORITY


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
    def __init__(self, kernels, override=True, priority="upward", adjusted=True, ilp=True):
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
        self.ilp = ilp #true = ilp plan, else fastest alone (core plan)
        self.T = None #ilp estimated makespan, none without the ilp
        self.solve_ms = None #how long making the plan took (one time cost)



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
        start = time.perf_counter()
        if self.ilp:
            self.plan, self.T = ilp_plan(self.kernels, self.cost, sim)
        else:
            self.plan = core_plan(self.kernels, self.cost)
        self.solve_ms = (time.perf_counter() - start) * 1e3
        self.rank = PRIORITY[self.priority](self.kernels, self.plan, self.cost, sim)
        self.n_desc = {kid: len(d) for kid, d in descendants(self.kernels).items()}
        