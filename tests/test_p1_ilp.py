import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator, fresh, topological
from sim.simulator import Simulator
from sim.workload import tree, fork_join, random_dag, fft, gaussian_elimination
from sim.policies.base import Policy
from sim.policies.p1_ilp import (ContentionILP, cost_table, descendants, core_plan, upward_rank,
                                 lower_bound_rank, eshcs_rank, PRIORITY)
from sim.policies.optimal import lower_bound


#same system as the other tests: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
def make_system(n_cores=2):
    return SystemConfig(
        cores=CoreConfig(n_cores=n_cores, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )


def run(kernels, policy, config=None):
    sim = Simulator(config or make_system(), policy)
    sim.submit(kernels)
    sim.run()
    return sim


families = [
    lambda: tree(seed=0),
    lambda: tree(inward=True, seed=1),
    lambda: fork_join(seed=0),
    lambda: random_dag(seed=0),
    lambda: fft(seed=0),
    lambda: gaussian_elimination(seed=0),
]


#a gemm that is faster on the accelerator goes there, an elementwise kernel can only go on the cpu
def test_core_plan():
    hw = Simulator(make_system(), Policy())
    ks = [make_gmm_kernel(0, 10, 32, 32, arrival=0.0), make_simple_kernel(1, 1000, arrival=0.0)]
    assert core_plan(ks, cost_table(ks, hw)) == {0: accelerator, 1: cpu}


#chain 0 -> 1 -> 2 of 400 ns kernels: the rank is the time left down the chain
def test_upward_rank_chain():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(1,))]
    cost = cost_table(ks, hw)
    rank = upward_rank(ks, core_plan(ks, cost), cost)
    assert rank == pytest.approx({0: 1200.0, 1: 800.0, 2: 400.0})


#1 core. kernel 0 is alone (rank 1200), kernel 1 has a long child (rank 3600) => kernel 1 goes first
#even though its listed 2nd, then its child (rank 3200) still beats kernel 0. order: 1, 2, 0
def test_critical_path_first():
    ks = [make_simple_kernel(0, 3000, arrival=0.0),                 #1200 ns, nothing after it
          make_simple_kernel(1, 1000, arrival=0.0),                 #400 ns, then
          make_simple_kernel(2, 8000, arrival=0.0, deps=(1,))]      #3200 ns
    run(ks, ContentionILP(ks), make_system(n_cores=1))
    assert ks[1].start == 0.0
    assert ks[2].start == pytest.approx(ks[1].finish)
    assert ks[0].start == pytest.approx(ks[2].finish)


#small gemm planned on the accelerator, but a big gemm holds it for ~2.6 us:
#the free cpu (410 ns) beats waiting => override puts it on the cpu, without override it waits
@pytest.mark.parametrize("override, where", [(True, cpu), (False, accelerator)])
def test_override(override, where):
    ks = [make_gmm_kernel(0, 64, 64, 64, arrival=0.0), make_gmm_kernel(1, 2, 32, 32, arrival=10.0)]
    run(ks, ContentionILP(ks, override=override))
    assert ks[0].resource == accelerator
    assert ks[1].resource == where
    if where == accelerator:
        assert ks[1].start == pytest.approx(ks[0].finish)


#without override every kernel runs where the plan put it
@pytest.mark.parametrize("make", families)
def test_follows_the_plan(make):
    ks = make()
    policy = ContentionILP(ks, override=False)
    sim = run(ks, policy)
    assert all(sim.runs[k.kernel_id].kind == policy.plan[k.kernel_id] for k in ks)


#every kernel finishes, after all its deps, and no schedule beats the lower bound, for every priority
@pytest.mark.parametrize("priority", list(PRIORITY))
@pytest.mark.parametrize("make", families)
def test_valid_schedule(make, priority):
    ks = make()
    sim = run(ks, ContentionILP(ks, priority=priority))
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks)
    for k in ks:
        assert all(k.start >= by_id[d].finish - 1e-9 for d in k.deps)
    assert sim.now >= lower_bound(fresh(ks), make_system()) * (1 - 1e-9)


#a gemm can use both resources, an elementwise kernel only the cpu
def test_cost_table():
    hw = Simulator(make_system(), Policy())
    ks = [make_gmm_kernel(0, 10, 32, 32, arrival=0.0), make_simple_kernel(1, 1000, arrival=0.0)]
    cost = cost_table(ks, hw)
    assert cost[0] == pytest.approx({cpu: 2048.0, accelerator: 332.8})
    assert cost[1] == pytest.approx({cpu: 400.0})


#diamond 0 -> (1, 2) -> 3: kernel 3 is reached twice from 0 but counted once
def test_descendants():
    ks = [make_simple_kernel(0, 1000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(3, 1000, arrival=0.0, deps=(1, 2))]
    assert descendants(ks) == {0: {1, 2, 3}, 1: {3}, 2: {3}, 3: set()}


#kernel 0 (400 ns) fans out to 6 memory bound kernels, kernel 7 starts a chain of 3 (1200 ns).
#upward rank only sees chain length => the chain goes first. lower bound sees that 0 and its
#children move 7 x 4000 bytes = 2800 ns at 10 GB/s => kernel 0 goes first
def test_lower_bound_rank_sees_traffic():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0)]
    ks += [make_simple_kernel(i, 1000, arrival=0.0, deps=(0,)) for i in range(1, 7)]
    ks += [make_simple_kernel(7, 1000, arrival=0.0),
           make_simple_kernel(8, 1000, arrival=0.0, deps=(7,)),
           make_simple_kernel(9, 1000, arrival=0.0, deps=(8,))]
    cost = cost_table(ks, hw)
    plan = core_plan(ks, cost)
    up = upward_rank(ks, plan, cost)
    lb = lower_bound_rank(ks, plan, cost, hw)
    assert up[0] == pytest.approx(800.0) and up[7] == pytest.approx(1200.0)
    assert lb[0] == pytest.approx(2800.0) and lb[7] == pytest.approx(1200.0)


#the lower bound rank is never below the upward rank (its 1st term). the ranks need topological order
@pytest.mark.parametrize("make", families)
def test_lower_bound_at_least_upward(make):
    hw = Simulator(make_system(), Policy())
    ks = topological(make())
    cost = cost_table(ks, hw)
    plan = core_plan(ks, cost)
    up = upward_rank(ks, plan, cost)
    lb = lower_bound_rank(ks, plan, cost, hw)
    assert all(lb[kid] >= up[kid] - 1e-9 for kid in up)


#chain 0 -> 1 -> 2 -> 3 next to a fan out 4 -> (5, 6, 7), all 400 ns. on the critical path urgency is 1,
#a chain kernel has 1 child out of max 3 so each level adds only a 1/3 of the one below it
def test_eshcs_rank():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0)]
    ks += [make_simple_kernel(i, 1000, arrival=0.0, deps=(i - 1,)) for i in range(1, 4)]
    ks += [make_simple_kernel(4, 1000, arrival=0.0)]
    ks += [make_simple_kernel(i, 1000, arrival=0.0, deps=(4,)) for i in range(5, 8)]
    cost = cost_table(ks, hw)
    rank = eshcs_rank(ks, core_plan(ks, cost), cost)
    assert rank[3] == pytest.approx(1.0)
    assert rank[0] == pytest.approx(1 + 1/3 + 1/9 + 1/27)
    assert rank[4] == pytest.approx(2/3)


#1 core. kernel 0 alone (800 ns) and kernel 1 (400 ns) + its child (400 ns) have the same upward rank,
#kernel 1 has a kernel after it => it goes first even though its listed 2nd
def test_tie_break_by_descendants():
    ks = [make_simple_kernel(0, 2000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(1,))]
    policy = ContentionILP(ks)
    run(ks, policy, make_system(n_cores=1))
    assert policy.rank[0] == pytest.approx(policy.rank[1])
    assert ks[1].start == 0.0
    assert ks[0].start == pytest.approx(ks[1].finish)


def test_unknown_priority():
    with pytest.raises(ValueError):
        ContentionILP([make_simple_kernel(0, 1000, arrival=0.0)], priority="fastest")