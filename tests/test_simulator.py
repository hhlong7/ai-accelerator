import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator
from sim.simulator import Simulator


#10 gflops per core, 32 x 32 array at 1 GHz, 10 GB/s dram => round numbers for hand traces
def make_system(dram=10e9):
    return SystemConfig(
        cores=CoreConfig(n_cores=2, gflops_per_core=10.0, multicore_speedup=0.5),
        accelerator=AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2),
        memory=MemoryConfig(dram_bw_bytes_per_s=dram),
    )


#test policy: each kernel goes where "where" says, in arrival order, waits if that resource is busy
class Fixed:
    def __init__(self, where):
        self.where = where      #{kernel_id: cpu or accelerator}
        self.waiting = []

    def on_queued(self, kernel, sim):
        self.waiting.append(kernel)

    def pick(self, sim):
        picks, still = [], []
        free_cores = len(sim.cpu.free_cores())
        accel_free = sim.accelerator.free
        for k in self.waiting:
            kind = self.where[k.kernel_id]
            if kind == cpu and free_cores >= 1:
                picks.append((k, cpu, 1))
                free_cores -= 1
            elif kind == accelerator and accel_free:
                picks.append((k, accelerator, None))
                accel_free = False
            else:
                still.append(k)
        self.waiting = still
        return picks


#memory bound cpu kernel: fpop 1000 => 100 ns compute, bytes 1000 * 2 * 2 = 4000 => wants 40 GB/s
#alone it gets 10 GB/s (peak) => 4000 / 10 GB/s = 400 ns
def mem_bound(kernel_id, arrival):
    return make_simple_kernel(kernel_id, 1000, arrival=arrival)


def run(kernels, where, **system):
    sim = Simulator(make_system(**system), Fixed(where))
    sim.submit(kernels)
    end = sim.run()
    return sim, end


def test_one_kernel_alone_takes_standalone_time():
    k = mem_bound(0, 0.00)
    sim, end = run([k], {0: cpu})
    assert k.start == 0.00
    assert k.finish == pytest.approx(400.0)
    assert end == pytest.approx(400.0)


#plan 1.4 validation: 2 identical memory bound kernels at the same time
#each wants 40 => capped at 10, total 20 > 10 => each gets 5 GB/s => 4000 / 5 GB/s = 800 ns = 2x standalone
def test_two_mem_bound_kernels_each_take_2x():
    k0, k1 = mem_bound(0, 0.00), mem_bound(1, 0.00)
    sim, end = run([k0, k1], {0: cpu, 1: cpu})
    assert k0.finish == pytest.approx(800.0)
    assert k1.finish == pytest.approx(800.0)


#plan 1.4 validation: when one finishes the other speeds up
#t=0..200   k0 alone, 10 GB/s => 200 / 400 = 50% done
#t=200      k1 arrives, both 5 GB/s (800 ns each)
#           k0: 50% left * 800 = 400 more => done at 600
#t=600      k1 did 400 / 800 = 50%, now alone at 10 GB/s => 50% * 400 = 200 more => done at 800
def test_other_kernel_speeds_up_after_one_finishes():
    k0, k1 = mem_bound(0, 0.00), mem_bound(1, 200.00)
    sim, end = run([k0, k1], {0: cpu, 1: cpu})
    assert k0.finish == pytest.approx(600.0)
    assert k1.start == pytest.approx(200.0)
    assert k1.finish == pytest.approx(800.0)
    assert end == pytest.approx(800.0)


#compute bound: 100 ops per element => fpop 100000 => 10000 ns compute, 4000 bytes => wants 0.4 GB/s
#2 of them want 0.8 GB/s total < 10 => no contention, both still 10000 ns
def test_compute_bound_kernels_dont_slow_each_other():
    k0 = make_simple_kernel(0, 1000, arrival=0.00, ops_per_element=100)
    k1 = make_simple_kernel(1, 1000, arrival=0.00, ops_per_element=100)
    sim, end = run([k0, k1], {0: cpu, 1: cpu})
    assert k0.finish == pytest.approx(10000.0)
    assert k1.finish == pytest.approx(10000.0)


#cpu vs accelerator contention, single tile gemm (10 x 32) @ (32 x 32):
#  alone: load 2048 bytes / 10 GB/s = 204.8 ns, compute 74 cycles but 1280 bytes / 10 GB/s = 128 ns => 332.8 ns
#together with the memory bound cpu kernel (both always capped at 10 GB/s => 5 GB/s each):
#t=0       cpu: 800 ns at 5 GB/s, accel load: 2048 / 5 GB/s = 409.6 ns
#t=409.6   load done. cpu did 409.6 / 800 = 51.2%, 48.8% left
#          accel compute: max(74, 1280 / 5 GB/s = 256) = 256 ns => done at 665.6
#t=665.6   accel done. cpu did 256 / 800 = 32% more => 16.8% left, alone at 10 GB/s => 0.168 * 400 = 67.2
#          => cpu done at 732.8
def test_cpu_and_accelerator_slow_each_other():
    k_cpu = mem_bound(0, 0.00)
    k_acc = make_gmm_kernel(1, 10, 32, 32, arrival=0.00)
    sim, end = run([k_cpu, k_acc], {0: cpu, 1: accelerator})

    assert sim.accelerator.standalone_ns(k_acc) == pytest.approx(332.8)
    assert k_acc.finish == pytest.approx(665.6) #2x its standalone
    assert k_cpu.finish == pytest.approx(732.8) #vs 400 standalone
    assert k_acc.resource == accelerator and k_cpu.resource == cpu


#accelerator runs 1 at a time: k1 waits for k0 then runs alone
def test_accelerator_one_at_a_time():
    k0 = make_gmm_kernel(0, 10, 32, 32, arrival=0.00)
    k1 = make_gmm_kernel(1, 10, 32, 32, arrival=0.00)
    sim, end = run([k0, k1], {0: accelerator, 1: accelerator})
    assert k0.finish == pytest.approx(332.8)
    assert k1.start == pytest.approx(332.8)
    assert k1.wait == pytest.approx(332.8)
    assert k1.finish == pytest.approx(665.6)


#the bandwidth used never goes over the peak, and the degradation hook from memory.py works:
#with 2 agents only 50% of the peak is left => 5 GB/s total => 2.5 each => 4000 / 2.5 GB/s = 1600 ns
def test_bw_never_over_peak_and_degradation_hook():
    k0, k1 = mem_bound(0, 0.00), mem_bound(1, 0.00)
    sim, end = run([k0, k1], {0: cpu, 1: cpu})
    assert all(bw <= 10e9 * (1 + 1e-9) for _, bw in sim.bw_log)

    half_when_shared = lambda n_agents: 1.0 if n_agents <= 1 else 0.5
    k2, k3 = mem_bound(2, 0.00), mem_bound(3, 0.00)
    sim = Simulator(make_system(), Fixed({2: cpu, 3: cpu}), degradation=half_when_shared)
    sim.submit([k2, k3])
    sim.run()
    assert k2.finish == pytest.approx(1600.0)


#a policy that never starts a kernel => run raises instead of finishing early with a wrong makespan
def test_stuck_kernels_raise():
    class Lazy:
        def on_queued(self, kernel, sim):
            pass

        def pick(self, sim):
            return []

    sim = Simulator(make_system(), Lazy())
    sim.submit([mem_bound(0, 0.00)])
    with pytest.raises(RuntimeError):
        sim.run()