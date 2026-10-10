"""
this is for the DRAM bandwidth model. the cpu cores and the accelerator pull from the same memory, so when
they want more than the peak bandwidth together they each get less => they slow down (contention).
allocate() takes how much each agent wants and returns how much each one gets.

how its shared (proportional share):
- an agent cant use more than the peak, so a demand over the peak (like inf from a load phase) counts as the peak
- if the total demand fits, everyone gets what they asked for
- if not, everyone is scaled down by the same factor so the total is the bandwidth available

degradation: on real hardware the total bandwidth drops when agents run together, more than simple sharing.
its a function n_agents => fraction of the peak thats left, and it will be the curve fitted to the
measurements of step 1.7. till then the default is no degradation
"""


def no_degradation(n_agents):
    return 1.0


#demands is {agent: bytes per sec it wants}, returns {agent: bytes per sec it gets}
def allocate(demands, memory, degradation=no_degradation):
    peak = memory.dram_bw_bytes_per_s
    for agent, demand in demands.items():
        if demand < 0:
            raise ValueError(f"Demand of {agent}: {demand} must be >= 0")

    allocation = {agent: 0.0 for agent in demands}
    active = {agent: min(demand, peak) for agent, demand in demands.items() if demand > 0}
    if not active:
        return allocation

    factor = degradation(len(active))
    if not 0 < factor <= 1:
        raise ValueError(f"Degradation factor {factor} for {len(active)} agents must be in (0, 1]")
    available = peak * factor

    total = sum(active.values())
    scale = min(1.0, available / total)
    for agent, demand in active.items():
        allocation[agent] = demand * scale
    return allocation
