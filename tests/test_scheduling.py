import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator, general_mat_mul_kernel, simple
from sim.simulator import Simulator
from sim.policies.fifo import Fifo
from sim.policies.heft_greedy import HeftGreedy
from sim.policies.qilin_style import QilinStyle
from sim.policies.contention_aware import ContentionAware


#same system as test_simulator.py: 2 cores at 10 gflops, 32 x 32 array at 1 GHz, 10 GB/s dram
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


#small gemm (1 x 32) @ (32 x 32), alone: 269.8 ns on the accelerator, 217.6 ns on a core
def small_gemm(kernel_id):
    return make_gmm_kernel(kernel_id, 1, 32, 32, arrival=0.00)


#workload 1: 3 big gemms then 2 memory bound kernels, all at t = 0
def gemm_heavy():
    return [big_gemm(0), big_gemm(1), big_gemm(2), mem_bound(3), mem_bound(4)]


#workload 2: 2 memory bound kernels, 2 small gemms, 1 more memory bound kernel, all at t = 0
def mixed():
    return [mem_bound(0), mem_bound(1), small_gemm(2), small_gemm(3), mem_bound(4)]


def run(policy, kernels):
    sim = Simulator(make_system(), policy)
    sim.submit(kernels)
    sim.run()
    return kernels


def where(kernels):
    return [k.resource for k in kernels]


#fifo takes the first free resource in arrival order:
#k0 => accelerator, k1 and k2 => the accelerator is busy so they take the 2 cores (about 6x slower there)
#k3 and k4 => nothing is free, they wait for the cores
def test_fifo_takes_first_free_resource():
    k = run(Fifo(), gemm_heavy())
    assert where(k) == [accelerator, cpu, cpu, cpu, cpu]
    assert k[1].start == 0.00 and k[2].start == 0.00
    assert k[3].start == pytest.approx(k[1].finish)
    assert k[4].start == pytest.approx(k[2].finish)


#heft picks the earliest finish with the standalone times, decisions at t = 0:
#k0: accelerator 332.8 vs core 2048 => accelerator
#k1: accelerator 332.8 + 332.8 = 665.6 vs core 2048 => waits for the accelerator
#k2: accelerator 665.6 + 332.8 = 998.4 vs core 2048 => waits for the accelerator
#k3 and k4: cpu only => 1 core each right away
def test_heft_waits_for_the_faster_resource():
    k = run(HeftGreedy(), gemm_heavy())
    assert where(k) == [accelerator, accelerator, accelerator, cpu, cpu]
    assert k[1].start == pytest.approx(k[0].finish)
    assert k[2].start == pytest.approx(k[1].finish)
    assert k[3].start == 0.00 and k[4].start == 0.00


#fifo on the mixed workload: k0, k1 => cores, k2 => accelerator, k3 is at the front with nothing free.
#the accelerator is free first (k2 is done at 757.2) so k3 goes there, k4 waits for a core
def test_fifo_mixed():
    k = run(Fifo(), mixed())
    assert where(k) == [cpu, cpu, accelerator, accelerator, cpu]
    assert k[2].finish == pytest.approx(757.2)
    assert k[3].start == pytest.approx(757.2)
    assert k[4].start == pytest.approx(k[0].finish)


#heft doesnt know the cores are slowed down by contention. at t = 757.2 the accelerator is free and k3 is waiting:
#heft thinks k0 and k1 were done at 400 (standalone) so a core is free "now": core 757.2 + 217.6 = 974.8
#                                                                 vs accelerator 757.2 + 269.8 = 1027.0
#=> it picks the core, but the cores are really busy till 1017.6 and the accelerator sits idle
def test_heft_is_blind_to_contention():
    k = run(HeftGreedy(), mixed())
    assert where(k) == [cpu, cpu, accelerator, cpu, cpu]
    assert k[2].finish == pytest.approx(757.2)
    assert k[0].finish == pytest.approx(1017.6)
    assert k[3].start == pytest.approx(1017.6) #waited 260.4 ns next to a free accelerator


#contention aware at the same moment, t = 757.2:
#it sees k0 and k1 are 32.55% from done, and with k2 gone they get 5 GB/s each => cores busy till
#757.2 + 0.3255 * 800 = 1017.6 (busy_until uses the fresh shares, not the old 4.55 GB/s)
#k3 on a core: 3 kernels share the bandwidth, 2176 bytes / 3.33 GB/s = 652.8 => 1017.6 + 652.8 = 1670.4
#k3 on the accelerator: load 614.4 + compute 142.8 = 757.2 => 757.2 + 757.2 = 1514.4
#=> it picks the accelerator and starts right away
def test_contention_aware_sees_the_contention():
    k = run(ContentionAware(), mixed())
    assert where(k) == [cpu, cpu, accelerator, accelerator, cpu]
    assert k[3].start == pytest.approx(757.2)
    assert k[4].start == pytest.approx(k[0].finish)


#no line till a resource has 2 finished kernels of different sizes => same decisions as heft
def test_qilin_without_history_is_heft():
    policy = QilinStyle()
    k = run(policy, gemm_heavy())
    assert where(k) == [accelerator, accelerator, accelerator, cpu, cpu]
    assert policy.fit == {} #every kernel of a type on a resource had the same size => no line yet


#2 points (fpop 1000, 500 ns) and (fpop 3000, 1500 ns) => line: time = 0.5 * fpop
def test_qilin_fits_a_line_to_history():
    policy = QilinStyle()
    sim = Simulator(make_system(), policy)
    new = make_simple_kernel(9, 2000, arrival=0.00)
    assert policy.estimate_ns(new, cpu, sim) == pytest.approx(800.0)    #no history: standalone, 8000 bytes / 10 GB/s

    for kernel_id, n_element, time in [(0, 1000, 500.0), (1, 3000, 1500.0)]:
        done = make_simple_kernel(kernel_id, n_element, arrival=0.00)
        done.resource, done.start, done.finish = cpu, 0.00, time
        policy.learn(done, sim)

    assert policy.estimate_ns(new, cpu, sim) == pytest.approx(1000.0)   #0.5 * 2000
    assert (cpu, simple) in policy.fit
    assert (accelerator, general_mat_mul_kernel) not in policy.fit  #the accelerator has no history yet


#the line from simple kernels is NOT used for a gemm on the same resource (1 line per kernel type)
#gemm (10 x 32) @ (32 x 32) on a core: still the standalone estimate 2048 ns, not 0.5 * fpop = 10240
def test_qilin_keeps_kernel_types_apart():
    policy = QilinStyle()
    sim = Simulator(make_system(), policy)
    for kernel_id, n_element, time in [(0, 1000, 500.0), (1, 3000, 1500.0)]:
        done = make_simple_kernel(kernel_id, n_element, arrival=0.00)
        done.resource, done.start, done.finish = cpu, 0.00, time
        policy.learn(done, sim)

    gemm = big_gemm(9)
    assert policy.estimate_ns(gemm, cpu, sim) == pytest.approx(2048.0)


#2 memory bound kernels at t = 0, each really takes 800 ns (test_simulator.py)
#k0 was predicted alone => 400, measured 800 => ratio 2.0 => correction 1.0 + 0.5 * (2.0 - 1.0) = 1.5
#k1 was predicted next to k0 => 800, measured 800 => ratio 1.0 => correction 1.5 + 0.5 * (1.0 - 1.5) = 1.25
def test_contention_aware_correction():
    policy = ContentionAware(alpha=0.5)
    k = run(policy, [mem_bound(0), mem_bound(1)])
    assert k[0].finish == pytest.approx(800.0)
    assert policy.correction[cpu] == pytest.approx(1.25)
    assert policy.correction[accelerator] == 1.0    #nothing ran there


#contention aware that remembers what busy_until and estimate_ns returned, to check the bug fixes
class RecordingContentionAware(ContentionAware):
    def __init__(self):
        super().__init__()
        self.busy = []      #(now, kernel_id, busy_until)
        self.raw = []       #(now, kernel_id, kind, prediction before correction)

    def busy_until(self, kernel, sim):
        time = super().busy_until(kernel, sim)
        self.busy.append((sim.now, kernel.kernel_id, time))
        return time

    def estimate_ns(self, kernel, kind, sim):
        time = super().estimate_ns(kernel, kind, sim)
        self.raw.append((sim.now, kernel.kernel_id, kind, self.predicted[kernel.kernel_id][kind]))
        return time


#bug fix: busy_until uses the fresh bandwidth shares. at t = 757.2 k2 just finished, pick runs before
#the simulator reallocates so k0 and k1 still have the old 4.55 GB/s share. they're 32.55% from done and
#will get 5 GB/s each => 757.2 + 0.3255 * 800 = 1017.6 (with the old share it said 1043.2)
def test_contention_aware_busy_until_uses_fresh_bandwidth():
    policy = RecordingContentionAware()
    run(policy, mixed())
    at_k2_finish = [time for now, kid, time in policy.busy if now == pytest.approx(757.2) and kid in (0, 1)]
    assert len(at_k2_finish) == 2
    assert all(time == pytest.approx(1017.6) for time in at_k2_finish)


#bug fix: a kernel finishing at the same time isn't a competitor. at t = 1147.8 k0 and k1 both finish,
#k1's event hasnt been processed yet when k4 is estimated. k4 only shares with k3 on the accelerator
#=> 5 GB/s => 4000 bytes / 5 GB/s = 800 (counting k1 too it said 1200)
def test_contention_aware_ignores_kernels_finishing_now():
    policy = RecordingContentionAware()
    run(policy, mixed())
    k4_on_cpu = [raw for now, kid, kind, raw in policy.raw if now == pytest.approx(1147.8) and kid == 4 and kind == cpu]
    assert k4_on_cpu and all(raw == pytest.approx(800.0) for raw in k4_on_cpu)


#every policy has to start and finish every kernel on both workloads
@pytest.mark.parametrize("make_policy", [Fifo, HeftGreedy, QilinStyle, ContentionAware])
@pytest.mark.parametrize("make_workload", [gemm_heavy, mixed])
def test_every_policy_finishes_everything(make_policy, make_workload):
    kernels = run(make_policy(), make_workload())
    assert all(k.done for k in kernels)
    assert all(k.start >= k.queue and k.finish > k.start for k in kernels)
