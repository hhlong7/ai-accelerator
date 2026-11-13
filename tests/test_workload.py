import math
import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import general_mat_mul_kernel, simple
from sim.simulator import Simulator
from sim.policies.heft_greedy import HeftGreedy
from sim.workload import (tree, fork_join, random_dag, fft, gaussian_elimination)


def make_system():
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )


#everything a kernel is made of, to compare 2 workloads
def fingerprint(kernels):
    return [(k.kernel_id, k.kernel_type, k.fpop, k.moved_bytes, k.deps, k.queue) for k in kernels]


#every dag generator with small sizes, used by the shared tests below
dags = {
    "tree": lambda **kw: tree(levels=3, fanout=2, **kw),
    "in_tree": lambda **kw: tree(levels=3, fanout=2, inward=True, **kw),
    "fork_join": lambda **kw: fork_join(width=3, stages=2, **kw),
    "random": lambda **kw: random_dag(n_nodes=12, n_levels=4, **kw),
    "fft": lambda **kw: fft(points=4, **kw),
    "ge": lambda **kw: gaussian_elimination(m=4, **kw),
}

#--- dag shapes, node counts by hand ---

def test_tree_size():
    assert len(tree(levels=4, fanout=2)) == 1 + 2 + 4 + 8
    assert len(tree(levels=3, fanout=3)) == 1 + 3 + 9


def test_tree_directions():
    out = tree(levels=2, fanout=2)
    assert [k.deps for k in out] == [(), (0,), (0,)]
    inward = tree(levels=2, fanout=2, inward=True)
    assert [k.deps for k in inward] == [(1, 2), (), ()]


def test_fork_join_shape():
    ks = fork_join(width=3, stages=2)
    assert len(ks) == 1 + 2 * (3 + 1)
    assert ks[4].deps == (1, 2, 3)      #1st join waits for the 1st 3 branches
    assert all(ks[i].deps == (4,) for i in (5, 6, 7))


#fft with 4 points: 7 recursive call nodes + 2 butterfly levels x 4 = 15
def test_fft_shape():
    ks = fft(points=4)
    assert len(ks) == 2 * 4 - 1 + 4 * int(math.log2(4))
    assert all(len(k.deps) == 2 for k in ks[7:])    #every butterfly node has 2 inputs
    with pytest.raises(ValueError):
        fft(points=6)


#(m^2 + m - 2) / 2 tasks
def test_gaussian_elimination_size():
    for m in (3, 5, 8):
        assert len(gaussian_elimination(m=m)) == (m * m + m - 2) // 2


def test_random_dag_every_node_past_the_1st_level_has_a_parent():
    ks = random_dag(n_nodes=30, n_levels=6, edge_prob=0.1, seed=3)
    roots = [k for k in ks if not k.deps]
    assert 1 <= len(roots) < len(ks)
    assert all(dep < k.kernel_id for k in ks for dep in k.deps)  #edges only go forward


#--- shared properties of every dag ---

@pytest.mark.parametrize("name", dags)
def test_same_seed_same_workload(name):
    assert fingerprint(dags[name](seed=7)) == fingerprint(dags[name](seed=7))


@pytest.mark.parametrize("name", dags)
def test_start_id_shifts_ids_and_deps(name):
    a, b = dags[name](seed=1), dags[name](seed=1, start_id=100)
    assert [k.kernel_id + 100 for k in a] == [k.kernel_id for k in b]
    assert [tuple(d + 100 for d in k.deps) for k in a] == [k.deps for k in b]


@pytest.mark.parametrize("name", dags)
def test_gemm_frac_controls_the_mix(name):
    assert all(k.kernel_type == simple for k in dags[name](gemm_frac=0.0))
    assert all(k.kernel_type == general_mat_mul_kernel for k in dags[name](gemm_frac=1.0))


#the real check: it runs in the simulator, no cycles, nothing starts before its deps finish
@pytest.mark.parametrize("name", dags)
def test_dag_runs_and_respects_deps(name):
    ks = dags[name](seed=2)
    sim = Simulator(make_system(), HeftGreedy())
    sim.submit(ks)
    sim.run()
    by_id = {k.kernel_id: k for k in ks}
    assert all(k.done for k in ks)
    assert all(k.start >= by_id[d].finish for k in ks for d in k.deps)


def test_two_workloads_together():
    a = fork_join(width=2, stages=1, seed=0)
    b = fft(points=4, seed=1, start_id=len(a))
    sim = Simulator(make_system(), HeftGreedy())
    sim.submit(a + b)
    sim.run()
    assert all(k.done for k in a + b)