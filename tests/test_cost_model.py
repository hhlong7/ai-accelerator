import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator
from sim.simulator import Simulator
from sim.policies.cost_model import what_if, play, Lane
from sim.model import Segment


#same system as test_metrics.py: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
def make_system():
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )


#mem bound: 100 ns compute, 4000 bytes => alone 400 ns, wants 40 GB/s
def mem_bound(kernel_id, arrival=0.0):
    return make_simple_kernel(kernel_id, 1000, arrival=arrival)


#compute bound: 100 ops per element => 10000 ns compute, 4000 bytes => wants 0.4 GB/s
def compute_bound(kernel_id, arrival=0.0):
    return make_simple_kernel(kernel_id, 1000, arrival=arrival, ops_per_element=100)


#big gemm: 332.8 ns alone on the accelerator
def big_gemm(kernel_id, arrival=0.0):
    return make_gmm_kernel(kernel_id, 10, 32, 32, arrival=arrival)


#starts kernels where it's told, and before starting one asks the cost model what will happen
class Probe:
    def __init__(self, where):
        self.where = where
        self.waiting = []
        self.guess = {} #kernel id -> (now, Estimate)

    def on_queued(self, kernel, sim):
        self.waiting.append(kernel)

    def pick(self, sim):
        picked, free_cores, accel_free = [], len(sim.cpu.free_cores()), sim.accelerator.free
        for k in list(self.waiting):
            kind = self.where[k.kernel_id]
            if kind == cpu and free_cores:
                free_cores -= 1
            elif kind == accelerator and accel_free:
                accel_free = False
            else:
                continue
            self.guess[k.kernel_id] = (sim.now, what_if(k, kind, sim, starting=picked))
            self.waiting.remove(k)
            picked.append((k, kind, 1))
        return picked


def run(kernels, where):
    probe = Probe(where)
    sim = Simulator(make_system(), probe)
    sim.submit(kernels)
    sim.run()
    return sim, probe.guess


#nothing running => the time alone, nobody to harm
def test_alone():
    sim, guess = run([mem_bound(0)], {0: cpu})
    now, est = guess[0]
    assert est.time == pytest.approx(400.0)
    assert est.harm == 0.0
    assert est.harm_by == {}


#k0 on the cpu, k1 starts at 200 (hand trace from test_simulator.py):
#without k1, k0 has 200 ns left. with k1 both get 5 GB/s => k0 done at 600 (400 from now, 200 extra)
#k1 then gets 10 GB/s => done at 800, 600 from now
def test_late_arrival_time_and_harm():
    sim, guess = run([mem_bound(0), mem_bound(1, arrival=200.0)], {0: cpu, 1: cpu})
    now, est = guess[1]
    assert now == 200.0
    assert est.time == pytest.approx(600.0)
    assert est.harm == pytest.approx(200.0)
    assert sim.kernels[1].finish == pytest.approx(now + est.time)


#2 kernels picked in the same pick: the 2nd one sees the 1st as running
def test_starting_counts_as_running():
    sim, guess = run([mem_bound(0), mem_bound(1)], {0: cpu, 1: cpu})
    assert guess[0][1].time == pytest.approx(400.0)     #k0 didnt know about k1 yet
    assert guess[1][1].time == pytest.approx(800.0)
    assert guess[1][1].harm == pytest.approx(400.0)     #k0: 400 => 800
    assert guess[1][1].harm_by == pytest.approx({0: 400.0})


#a compute bound kernel barely needs bandwidth => much less harm than a memory bound one
def test_compute_bound_harms_less():
    _, mem = run([mem_bound(0), mem_bound(1, arrival=100.0)], {0: cpu, 1: cpu})
    _, comp = run([mem_bound(0), compute_bound(1, arrival=100.0)], {0: cpu, 1: cpu})
    assert comp[1][1].harm < 0.05 * mem[1][1].harm


#the prediction matches the simulator when nothing else starts later: every kernel's finish
@pytest.mark.parametrize("make, where", [
    (lambda: [mem_bound(0), big_gemm(1, arrival=50.0)], {0: cpu, 1: accelerator}),
    (lambda: [big_gemm(0), mem_bound(1, arrival=120.0)], {0: accelerator, 1: cpu}),
    (lambda: [mem_bound(0), compute_bound(1, arrival=30.0)], {0: cpu, 1: cpu}),
    (lambda: [big_gemm(0), compute_bound(1, arrival=10.0), mem_bound(2, arrival=60.0)], {0: accelerator, 1: cpu, 2: cpu}),
])
def test_matches_the_simulator(make, where):
    ks = make()
    sim, guess = run(ks, where)
    last = ks[-1]
    now, est = guess[last.kernel_id]
    assert last.finish == pytest.approx(now + est.time)
    #harm = how much later the others finished than if the last one had never come
    alone_sim, _ = run(make()[:-1], where)
    extra = {k.kernel_id: k.finish - a.finish for k, a in zip(ks[:-1], alone_sim.kernels)}
    assert est.harm_by == pytest.approx(extra)
    assert est.harm == pytest.approx(sum(extra.values()))


#play() on its own: 2 lanes that want the full bandwidth share it, 1 GB/s system for round numbers
def test_play_two_lanes():
    memory = MemoryConfig(dram_bw_bytes_per_s=1e9)
    lanes = {0: Lane([Segment(0.0, 100)]), 1: Lane([Segment(0.0, 300)])}
    finish = play(lanes, memory, lambda n: 1.0)
    #both at 0.5 GB/s till 0 is done at 200 (100 bytes), 1 has 200 bytes left alone => +200
    assert finish == pytest.approx({0: 200.0, 1: 400.0})
    assert lanes[0].left == 1.0     #the caller's lanes are not changed