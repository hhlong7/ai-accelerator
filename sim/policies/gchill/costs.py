"""
costs for the new algorithm (step 2a): how long each kernel takes on each resource
- cost_table: alone, with the whole dram bandwidth
- adjusted_cost_table: with the dram sharing it should expect from the kernels next to it
every plan and rank reads its times from one of these tables
"""

from sim.kernels import cpu, accelerator
from sim.estimate import fits_accel, standalone_ns
from sim.memory import allocate
from sim.policies.cost_model import segments
from sim.policies.gchill.plan import core_plan
from sim.policies.gchill.priority import upward_rank


#{kernel id: {resource: ns alone}}, only the resources the kernel can run on. every rank and the
#plan read their times from here, so swapping in contention-adjusted times later changes all of them
def cost_table(kernels, sim):
    cost = {}
    for k in kernels:
        cost[k.kernel_id] = {cpu: standalone_ns(k, cpu, sim)}
        if fits_accel(k, sim):
            cost[k.kernel_id][accelerator] = standalone_ns(k, accelerator, sim)
    return cost


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
