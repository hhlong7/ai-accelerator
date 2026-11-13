import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator, chain
from sim.simulator import Simulator
from sim.policies.optimal import Replay


#same system as test_scheduling.py: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
def make_system():
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )


#memory bound cpu only kernel, alone: 400 ns on a core
def mem_bound(kernel_id):
    return make_simple_kernel(kernel_id, 1000, arrival=0.00)


#big gemm (10 x 32) @ (32 x 32), alone: 332.8 ns on the accelerator, 2048 ns on a core
def big_gemm(kernel_id):
    return make_gmm_kernel(kernel_id, 10, 32, 32, arrival=0.00)


def run(kernels, order, place):
    sim = Simulator(make_system(), Replay(order, place))
    sim.submit(kernels)
    return sim.run()


#every kernel runs where the plan says, with the cores the plan says
def test_replay_follows_the_placement():
    ks = [big_gemm(0), big_gemm(1), mem_bound(2)]
    run(ks, [0, 1, 2], {0: (accelerator, None), 1: (cpu, 2), 2: (cpu, 1)})
    assert [k.resource for k in ks] == [accelerator, cpu, cpu]
    assert ks[0].start == ks[1].start == 0.0
    #kernel 1 took both cores, so kernel 2 waits for it
    assert ks[2].start == pytest.approx(ks[1].finish)


#kernel 2 has free cores at t = 0 but its behind kernel 1, which waits for the accelerator
def test_replay_keeps_the_start_order():
    ks = [big_gemm(0), big_gemm(1), mem_bound(2)]
    run(ks, [0, 1, 2], {0: (accelerator, None), 1: (accelerator, None), 2: (cpu, 1)})
    assert ks[0].start == 0.0
    assert ks[1].start == pytest.approx(ks[0].finish)
    assert ks[2].start == pytest.approx(ks[1].start)


#same kernels, other order => kernel 2 goes first and starts at t = 0
def test_replay_other_order_other_schedule():
    ks = [big_gemm(0), big_gemm(1), mem_bound(2)]
    run(ks, [2, 0, 1], {0: (accelerator, None), 1: (accelerator, None), 2: (cpu, 1)})
    assert ks[2].start == ks[0].start == 0.0


#a kernel never starts before its deps finish
def test_replay_waits_for_deps():
    ks = chain([big_gemm(0), mem_bound(1)])
    run(ks, [0, 1], {0: (accelerator, None), 1: (cpu, 1)})
    assert ks[1].start == pytest.approx(ks[0].finish)
