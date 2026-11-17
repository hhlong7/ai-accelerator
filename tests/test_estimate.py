import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator
from sim.policies.base import Policy
from sim.simulator import Simulator
from sim.estimate import fits_accel, standalone_ns, fastest_ns


#same system as test_scheduling.py: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
def make_sim():
    config = SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )
    return Simulator(config, Policy())


#memory bound: 1000 fpop = 100 ns of compute, but 4000 bytes / 10 GB/s = 400 ns
def mem_bound():
    return make_simple_kernel(0, 1000, arrival=0.00)


#(10 x 32) @ (32 x 32): 20480 fpop = 2048 ns on a core, 332.8 ns on the accelerator
def big_gemm():
    return make_gmm_kernel(1, 10, 32, 32, arrival=0.00)


#(1 x 32) @ (32 x 32): faster on a core (217.6 ns) than on the accelerator (269.8 ns)
def small_gemm():
    return make_gmm_kernel(2, 1, 32, 32, arrival=0.00)


#the memory time counts, cores.py only counts the 100 ns of compute
def test_memory_bound_kernel_is_limited_by_bandwidth():
    sim = make_sim()
    assert sim.cpu.standalone_ns(mem_bound()) == pytest.approx(100.0)
    assert standalone_ns(mem_bound(), cpu, sim) == pytest.approx(400.0)


def test_gemm_on_each_resource():
    sim = make_sim()
    assert standalone_ns(big_gemm(), cpu, sim) == pytest.approx(2048.0)
    assert standalone_ns(big_gemm(), accelerator, sim) == pytest.approx(332.8)
    #2 cores at multicore_speedup 0.5 => 1.5x faster
    assert standalone_ns(big_gemm(), cpu, sim, num_cores=2) == pytest.approx(2048.0 / 1.5)


def test_fits_accel():
    sim = make_sim()
    assert fits_accel(big_gemm(), sim)
    assert not fits_accel(mem_bound(), sim)


#fastest picks the best resource it is allowed on
def test_fastest():
    sim = make_sim()
    assert fastest_ns(big_gemm(), sim) == pytest.approx(332.8)
    assert fastest_ns(small_gemm(), sim) == pytest.approx(217.6)
    assert fastest_ns(mem_bound(), sim) == pytest.approx(400.0) #cpu only
    #a 3 core option doesnt exist on a 2 core cpu, so its skipped
    assert fastest_ns(mem_bound(), sim, core_options=(1, 3)) == pytest.approx(400.0)
