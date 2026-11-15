import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator
from sim.simulator import Simulator
from sim.policies.heft_greedy import HeftGreedy
from sim.workload import fork_join, fft
from sim import metrics


#same system as test_simulator.py: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
def make_system():
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=10e9),
    )


#runs every kernel exactly where it's told
class Fixed:
    def __init__(self, where):
        self.where = where
        self.waiting = []

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
            self.waiting.remove(k)
            picked.append((k, kind, 1))
        return picked


def run(kernels, where):
    sim = Simulator(make_system(), Fixed(where))
    sim.submit(kernels)
    sim.run()
    return sim


#mem bound kernel: 1000 fpop = 100 ns compute, 4000 bytes = 400 ns at 10 GB/s => alone 400 ns
def mem_bound(kernel_id, arrival):
    return make_simple_kernel(kernel_id, 1000, arrival=arrival)


#2 mem bound kernels together at t = 0: 5 GB/s each => both 0 => 800
def test_two_kernels_sharing_bandwidth():
    sim = run([mem_bound(0, 0.0), mem_bound(1, 0.0)], {0: cpu, 1: cpu})
    assert metrics.makespan(sim) == pytest.approx(800.0)
    assert metrics.throughput(sim) == pytest.approx(2 / 800e-9)
    assert metrics.latency(sim)["mean"] == pytest.approx(800.0)
    assert metrics.cpu_utilization(sim) == pytest.approx(1.0)      #both cores busy the whole time
    assert metrics.accelerator_utilization(sim) == pytest.approx(0.0)
    assert metrics.dram_utilization(sim) == pytest.approx(1.0)     #10 GB/s used the whole time
    assert metrics.bytes_moved(sim) == pytest.approx(8000)
    assert metrics.slowdowns(sim) == pytest.approx({0: 2.0, 1: 2.0})


#k0 at 0, k1 at 200 (hand trace from test_simulator.py):
#0..200 k0 alone at 10 GB/s => 2000 of 4000 bytes. 200..600 both at 5 GB/s => k0 done at 600
#k1 has 2000 bytes done at 600, the rest alone at 10 GB/s => done at 800
def test_late_arrival():
    sim = run([mem_bound(0, 0.0), mem_bound(1, 200.0)], {0: cpu, 1: cpu})
    k0, k1 = sim.kernels
    assert (k0.finish, k1.finish) == (pytest.approx(600.0), pytest.approx(800.0))
    assert metrics.makespan(sim) == pytest.approx(800.0)
    assert metrics.latency(sim)["mean"] == pytest.approx(600.0)    #both 600
    assert metrics.cpu_utilization(sim) == pytest.approx((600 + 600) / (2 * 800))
    assert metrics.dram_utilization(sim) == pytest.approx(1.0)
    assert metrics.slowdowns(sim) == pytest.approx({0: 1.5, 1: 1.5})


#big gemm alone on the accelerator: 332.8 ns, no contention
def test_accelerator_alone():
    k = make_gmm_kernel(0, 10, 32, 32, arrival=0.0)
    sim = run([k], {0: accelerator})
    assert metrics.makespan(sim) == pytest.approx(332.8)
    assert metrics.accelerator_utilization(sim) == pytest.approx(1.0)
    assert metrics.cpu_utilization(sim) == pytest.approx(0.0)
    assert metrics.slowdowns(sim)[0] == pytest.approx(1.0)
    assert sim.runs[0].kind == accelerator and sim.runs[0].num_cores == 0


#makespan starts at the 1st arrival, not at 0
def test_makespan_starts_at_first_arrival():
    sim = run([mem_bound(0, 1000.0)], {0: cpu})
    assert metrics.makespan(sim) == pytest.approx(400.0)


def test_percentile_nearest_rank():
    values = list(range(1, 101))    #1..100
    assert metrics.percentile(values, 50) == 50
    assert metrics.percentile(values, 95) == 95
    assert metrics.percentile(values, 99) == 99
    assert metrics.percentile(values, 100) == 100
    assert metrics.percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        metrics.percentile([], 50)


def test_gap():
    assert metrics.gap(125.0, 100.0) == pytest.approx(0.25)
    assert metrics.gap(100.0, 100.0) == 0.0
    with pytest.raises(ValueError):
        metrics.gap(1.0, 0.0)


#on a real dag: bytes from the bandwidth log = bytes the kernels' segments move, everything in range
@pytest.mark.parametrize("make", [lambda: fork_join(width=3, stages=2, seed=1), lambda: fft(points=4, seed=2)])
def test_dag_run_is_consistent(make):
    ks = make()
    sim = Simulator(make_system(), HeftGreedy())
    sim.submit(ks)
    sim.run()
    assert metrics.bytes_moved(sim) == pytest.approx(sum(r.bytes for r in sim.runs.values()))
    row = metrics.summary(sim)
    assert row["n_kernels"] == len(ks)
    for key in ("cpu_util", "accel_util", "dram_util"):
        assert 0.0 <= row[key] <= 1.0 + 1e-9
    assert row["slowdown_mean"] >= 1.0 - 1e-9
    assert row["latency_mean_ns"] <= row["latency_p95_ns"] <= row["latency_p99_ns"] <= row["makespan_ns"]
    assert sim.n_picks > 0 and row["sched_total_ns"] > 0


def test_summary_needs_a_finished_run():
    sim = Simulator(make_system(), HeftGreedy())
    sim.submit([mem_bound(0, 0.0)])
    with pytest.raises(ValueError):
        metrics.summary(sim)