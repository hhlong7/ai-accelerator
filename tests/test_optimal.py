import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator, chain
from sim.simulator import Simulator
from sim.kernels import fresh
from sim.workload import fork_join, tree, random_dag
from sim.policies.base import Policy
from sim.policies.fifo import Fifo
from sim.policies.heft_greedy import HeftGreedy
from sim.policies.qilin_style import QilinStyle
from sim.policies.contention_aware import ContentionAware
from sim.policies.optimal import Replay, lower_bounds, lower_bound, start_orders, placements, best_schedule



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




#part 2 and 3

#same cores and array, 50 GB/s: mem_bound is now compute bound alone (100 ns compute > 80 ns memory)
def fast_memory():
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=50e9),
    )


#chain of 3 at 100 ns each: the chain is the bound, and running it reaches it
def test_lower_bound_critical_path():
    ks = chain([mem_bound(0), mem_bound(1), mem_bound(2)])
    lb = lower_bounds(ks, fast_memory())
    assert lb["critical_path"] == pytest.approx(300.0)
    assert lb["bytes"] == pytest.approx(240.0) #12000 bytes / 50 GB/s
    assert lower_bound(ks, fast_memory()) == pytest.approx(300.0)
    sim = Simulator(fast_memory(), Fifo())
    sim.submit(ks)
    assert sim.run() == pytest.approx(300.0)


#6 independent at 100 ns each on 2 cores: cpu_work = 300, but 2 at once want 80 GB/s and only
#get 50, so dram is the real limit: 24000 bytes / 50 GB/s = 480, and fifo hits exactly that
def test_lower_bound_bytes_and_cpu_work():
    ks = [mem_bound(i) for i in range(6)]
    lb = lower_bounds(ks, fast_memory())
    assert lb["cpu_work"] == pytest.approx(300.0)
    assert lb["work"] == pytest.approx(200.0) #600 / 3, the accelerator doesnt help them
    assert lb["bytes"] == pytest.approx(480.0)
    sim = Simulator(fast_memory(), Fifo())
    sim.submit(ks)
    assert sim.run() == pytest.approx(480.0)


#diamond 0 -> {1, 2} -> 3 has 2 orders, 0 and 3 are fixed
def test_start_orders_respect_deps():
    ks = [mem_bound(0), mem_bound(1), mem_bound(2), mem_bound(3)]
    ks[1].deps, ks[2].deps, ks[3].deps = (0,), (0,), (1, 2)
    assert sorted(start_orders(ks)) == [[0, 1, 2, 3], [0, 2, 1, 3]]
    assert len(list(start_orders([mem_bound(i) for i in range(3)]))) == 6


#gemm: cpu or accelerator, mem_bound: cpu only => 2 placements
def test_placements():
    hw = Simulator(make_system(), Policy())
    places = list(placements([big_gemm(0), mem_bound(1)], hw))
    assert places == [{0: (cpu, 1), 1: (cpu, 1)}, {0: (accelerator, None), 1: (cpu, 1)}]


#2 big gemms at 10 GB/s: both on the accelerator, one after the other = 665.6 ns,
#which is also the bytes bound (6656 bytes / 10 GB/s)
def test_best_schedule_small_case():
    ks = [big_gemm(0), big_gemm(1)]
    end, order, place = best_schedule(ks, make_system())
    assert end == pytest.approx(665.6)
    assert place == {0: (accelerator, None), 1: (accelerator, None)}
    assert end == pytest.approx(lower_bound(ks, make_system()))


#replaying the plan it returns gives the same makespan
def test_best_schedule_replays():
    ks = fork_join(width=2, stages=1, seed=3)
    end, order, place = best_schedule(ks, make_system())
    sim = Simulator(make_system(), Replay(order, place))
    sim.submit(fresh(ks))
    assert sim.run() == pytest.approx(end)


def test_best_schedule_refuses_big_dags():
    with pytest.raises(ValueError):
        best_schedule([mem_bound(i) for i in range(9)], make_system())


#the check from the plan (1.8): the best schedule is never worse than any policy,
#and never better than the lower bound
@pytest.mark.parametrize("workload", [
    lambda: fork_join(width=2, stages=1, seed=1),
    lambda: fork_join(width=3, stages=1, seed=2),
    lambda: tree(levels=3, fanout=2, seed=4),
    lambda: tree(levels=3, fanout=2, inward=True, seed=5),
    lambda: random_dag(n_nodes=6, n_levels=3, edge_prob=0.5, seed=6),
])
@pytest.mark.parametrize("config", [make_system, fast_memory])
def test_best_schedule_beats_every_policy(workload, config):
    ks = workload()
    best, _, _ = best_schedule(ks, config())
    assert best >= lower_bound(ks, config()) * (1 - 1e-9)
    for policy in [Fifo, HeftGreedy, QilinStyle, ContentionAware]:
        sim = Simulator(config(), policy())
        sim.submit(fresh(ks))
        assert best <= sim.run() * (1 + 1e-9), policy.__name__
