"""
estimate.py figures out for every 
policy and for optimal.py: how long does a 
kernel take on a resource if it runs alone 
with the whole dram bandwidth 
(its standalone time)?

it runs the same segments the simulator runs (model.py), 
each one at the peak bandwidth. so a
memory bound kernel is just as slow here as it
is in the simulator when nothing else is 
running. dont use cpu.standalone_ns from 
cores.py as a run time, it only counts the 
compute.

every function takes sim = anything with .cpu,
.accelerator and .config, like a Simulator

"""

from sim.kernels import cpu, accelerator
from sim.model import cpu_segs, accelerator_segs


# can this kernel go on the accelerator (gemm only, and same dtype as the array)
def fits_accel(kernel, sim):
    return kernel.can_run_on(accelerator) and kernel.dtype_bytes == sim.config.accelerator.dtype_bytes


# the segments the sim would run for this kernel
# on this resource
def segments(kernel, kind, sim, num_cores = 1):
    if kind == cpu:
        return cpu_segs(kernel, sim.cpu, num_cores)
    if kind == accelerator:
        return accelerator_segs(kernel, sim.accelerator)
    raise ValueError(f"Unknown resource kind: {kind}")


# time alone with full dram bandwidth
def standalone_ns(kernel, kind, sim, num_cores = 1):
    peak = sim.config.memory.dram_bw_bytes_per_s

    return sum(s.duration_ns(peak) for s in segments(kernel, kind, sim ,num_cores))


# the fastest kernel can run over the accel
# and every core count is might use
def fastest_ns(kernel, sim, core_options = (1,)):
    n_cores = sim.config.cores.n_cores
    times = [standalone_ns(kernel, cpu, sim, n) for n in core_options
             if n <= n_cores and (n == 1 or kernel.parallelize)]
    if fits_accel(kernel, sim):
        times.append(standalone_ns(kernel, accelerator, sim))

    return min(times)