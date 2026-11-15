"""
metrics is where we get the numbers we reported for one finished run, reading from the simulator
after a sim.run()

the following functions are;
-makespan: first arrival to last finish (ns)
-throughput: kernels finished per second
-latency: finish - arrival per kernel => mean, p95, p99
-utilization: how busy the cpu cores are the accelerator were over the makespan
-dram util: avg bw used / peak
-slowdown: how much slower a kernel ran than alone with the full bw, if 1 then means no contention
-gap: how far a makespan is above a reference (optimal schedule or a lower bound)
-scheduler time: wall clock time spent inside the policy, total and per pick

"""

import math
from sim.kernels import cpu, accelerator

def makespan(sim):
    return max(k.finish for k in sim.kernels) - min(k.queue for k in sim.kernels)

def throughput(sim):
    return len(sim.kernels) / makespan(sim) * 1e9

def percentile(val, p): #nearest rank p: the smallest val with at least p% of the values <= it
    if not val:
        raise ValueError("percentile of an empty list")
    if not 0 < p <= 100:
        raise ValueError(f"p: {p} must be between 0 to 100")
    ordered = sorted(val)
    return ordered[math.ceil(p / 100 * len(ordered)) - 1]

def latency(sim):
    lat = [k.latency for k in sim.kernels]
    return {"mean": sum(lat) / len(lat), "p95": percentile(lat, 95), "p99": percentile(lat, 99)}

def cpu_utilization(sim):
    busy = sum(sim.runs[k.kernel_id].num_cores * (k.finish - k.start) for k in sim.kernels if k.resource == cpu)
    return busy / (sim.config.cores.n_cores * makespan(sim))

def accelerator_utilization(sim):
    busy = sum(k.finish - k.start for k in sim.kernels if k.resource == accelerator)
    return busy / makespan(sim)

def bytes_moved(sim): #d/dt bytes moved = bw over t
    log = sim.bw_log
    return sum(bw * (t_next - t) / 1e9 for (t, bw), (t_next, _) in zip(log, log[1:]))

def dram_utilization(sim):
    peak_bytes = sim.config.memory.dram_bw_bytes_per_s * makespan(sim) / 1e9
    return bytes_moved(sim) / peak_bytes

def slowdowns(sim):
    return {k.kernel_id: (k.finish - k.start) / sim.runs[k.kernel_id].alone_ns for k in sim.kernels}

def gap(value, reference):
    if reference <= 0:
        raise ValueError(f"reference : {reference} must be positive")
    return (value - reference) / reference

def scheduler_time(sim):
    return {"total_ns": sim.sched_ns, "per_pick_ns": sim.sched_ns / sim.n_picks if sim.n_picks else 0.0}

def summary(sim):
    if not sim.kernels or not all(k.done for k in sim.kernels):
        raise ValueError("summary needs a finished run")
    lat, slow, sched = latency(sim), list(slowdowns(sim).values()), scheduler_time(sim)
    return {
        "n_kernels": len(sim.kernels),
        "makespan_ns": makespan(sim),
        "throughput_per_s": throughput(sim),
        "latency_mean_ns": lat["mean"],
        "latency_p95_ns": lat["p95"],
        "latency_p99_ns": lat["p99"],
        "cpu_util": cpu_utilization(sim),
        "accel_util": accelerator_utilization(sim),
        "dram_util": dram_utilization(sim),
        "slowdown_mean": sum(slow) / len(slow),
        "slowdown_max": max(slow),
        "sched_total_ns": sched["total_ns"],
        "sched_per_pick_ns": sched["per_pick_ns"],
    }
