import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator, fresh
from sim.simulator import Simulator
from sim.workload import tree, fork_join, random_dag, fft, gaussian_elimination
from sim.policies.base import Policy
from sim.policies.p1_ilp import ContentionILP, core_plan, upward_rank
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
    assert core_plan(ks, hw) == {0: accelerator, 1: cpu}


#chain 0 -> 1 -> 2 of 400 ns kernels: the rank is the time left down the chain
def test_upward_rank_chain():
    hw = Simulator(make_system(), Policy())
    ks = [make_simple_kernel(0, 1000, arrival=0.0),
          make_simple_kernel(1, 1000, arrival=0.0, deps=(0,)),
          make_simple_kernel(2, 1000, arrival=0.0, deps=(1,))]
    rank = upward_rank(ks, core_plan(ks, hw), hw)
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


#every kernel finishes, after all its deps, and no schedule beats the lower bound
@pytest.mark.parametrize("make", families)
def test_valid_schedule(make):
    ks = make()
    sim = run(ks, ContentionILP(ks))
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks)
    for k in ks:
        assert all(k.start >= by_id[d].finish - 1e-9 for d in k.deps)
    assert sim.now >= lower_bound(fresh(ks), make_system()) * (1 - 1e-9)