"""
runs the same policies on 3 versions of the workload to show when they differ:
1. original mixed workload at 10 GB/s => all memory bound, dram saturated => every policy gets the same makespan
2. + 2 compute bound gemms at 10 GB/s => policies start to differ
3. + 2 compute bound gemms at 50 GB/s => dram not the bottleneck anymore
run from the repo root: python -m experiments.run_variants
"""

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu
from sim.simulator import Simulator
from sim.policies.fifo import Fifo
from sim.policies.heft_greedy import HeftGreedy
from sim.policies.qilin_style import QilinStyle
from sim.policies.contention_aware import ContentionAware


policies = [Fifo, HeftGreedy, QilinStyle, ContentionAware]


#same system as the tests, only the dram bandwidth changes
def make_system(dram_bw):
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=dram_bw),
    )


#the mixed workload from test_scheduling.py: all memory bound
def base():
    return [
        make_simple_kernel(0, 1000, arrival=0.00),   #memory bound, cpu only
        make_simple_kernel(1, 1000, arrival=0.00),
        make_gmm_kernel(2, 1, 32, 32, arrival=0.00), #small gemm
        make_gmm_kernel(3, 1, 32, 32, arrival=0.00),
        make_simple_kernel(4, 1000, arrival=0.00),
    ]


#base + 2 compute bound gemms (64 x 128) @ (128 x 128)
def with_big_gemms():
    return base() + [
        make_gmm_kernel(5, 64, 128, 128, arrival=0.00),
        make_gmm_kernel(6, 64, 128, 128, arrival=0.00),
    ]


#(label, workload function, dram bandwidth)
variants = [
    ("original, 10 GB/s", base, 10e9),
    ("+2 compute bound gemms, 10 GB/s", with_big_gemms, 10e9),
    ("+2 compute bound gemms, 50 GB/s", with_big_gemms, 50e9),
]


for label, workload, dram_bw in variants:
    #lower bound on makespan: all the bytes moved at full dram speed
    total_bytes = sum(k.moved_bytes for k in workload())
    print(f"--- {label}  (total bytes / bw = {total_bytes / dram_bw * 1e9:.1f} ns)")

    for policy in policies:
        kernels = workload() #new kernels every run, the simulator fills in start/finish
        sim = Simulator(make_system(dram_bw), policy())
        sim.submit(kernels)
        end = sim.run()

        where = ["C" if k.resource == cpu else "A" for k in kernels]
        mean_latency = sum(k.latency for k in kernels) / len(kernels)
        print(f"  {policy.__name__:16s} makespan {end:9.1f} ns   placement {where}   mean latency {mean_latency:9.1f} ns")
    print()


"""
--- original, 10 GB/s  (total bytes / bw = 1635.2 ns)
  Fifo             makespan    1635.2 ns   placement ['C', 'C', 'A', 'A', 'C']   mean latency    1212.6 ns
  HeftGreedy       makespan    1635.2 ns   placement ['C', 'C', 'A', 'C', 'C']   mean latency    1176.1 ns
  QilinStyle       makespan    1635.2 ns   placement ['C', 'C', 'A', 'C', 'C']   mean latency    1176.1 ns
  ContentionAware  makespan    1635.2 ns   placement ['C', 'C', 'A', 'A', 'C']   mean latency    1212.6 ns

--- +2 compute bound gemms, 10 GB/s  (total bytes / bw = 14742.4 ns)
  Fifo             makespan  211420.0 ns   placement ['C', 'C', 'A', 'A', 'C', 'C', 'A']   mean latency   32472.9 ns
  HeftGreedy       makespan   17046.4 ns   placement ['C', 'C', 'A', 'C', 'C', 'A', 'A']   mean latency    4772.3 ns
  QilinStyle       makespan  211892.1 ns   placement ['C', 'C', 'A', 'C', 'C', 'A', 'C']   mean latency   32635.9 ns
  ContentionAware  makespan   17046.4 ns   placement ['C', 'C', 'A', 'A', 'C', 'A', 'A']   mean latency    4672.9 ns

--- +2 compute bound gemms, 50 GB/s  (total bytes / bw = 2948.5 ns)
  Fifo             makespan  210106.1 ns   placement ['C', 'C', 'A', 'C', 'C', 'A', 'C']   mean latency   30661.3 ns
  HeftGreedy       makespan    5826.4 ns   placement ['C', 'C', 'A', 'A', 'C', 'A', 'A']   mean latency    1459.1 ns
  QilinStyle       makespan    5826.4 ns   placement ['C', 'C', 'A', 'A', 'C', 'A', 'A']   mean latency    1459.1 ns
  ContentionAware  makespan    5826.4 ns   placement ['C', 'C', 'A', 'A', 'C', 'A', 'A']   mean latency    1459.1 ns
  
"""