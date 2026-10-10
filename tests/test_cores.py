import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig
from sim.engine import Engine, queued, finished
from sim.kernels import make_gmm_kernel, make_simple_kernel, cpu, accelerator
from sim.cores import CPU, Accelerator


#10 gflops per core => a simple kernel with n elements (fpop = n) takes n / 10 ns
def make_cpu(num_cores=2):
    config_core = CoreConfig(n_cores=num_cores, gflops_per_core=10.0, multicore_speedup=0.5)
    return CPU(config_core)


def make_accel():
    config_accel = AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2)
    config_mem = MemoryConfig(dram_bw_bytes_per_s=50e9)
    return Accelerator(config_accel, config_mem)


#fpop = 100 => 100 / 10 = 10 ns on 1 core
#speedup: 2 cores = 1 + 1*0.5 = 1.5x, 4 cores = 1 + 3*0.5 = 2.5x
def test_cpu_standalone_time():
    c = make_cpu(num_cores=4)
    k = make_simple_kernel(0, 100, arrival=0.00)
    assert c.standalone_ns(k) == pytest.approx(10.00)
    assert c.speedup(2) == pytest.approx(1.50)
    assert c.standalone_ns(k, num_cores=2) == pytest.approx(100 / 15)
    assert c.standalone_ns(k, num_cores=4) == pytest.approx(4.00)


def test_cpu_assign_and_release():
    c = make_cpu()
    k = make_simple_kernel(0, 100, arrival=0.00)

    time = c.assign(k, now=5.00)
    assert time == pytest.approx(10.00)
    assert k.resource == cpu and k.start == 5.00
    assert not c.cores[0].free and c.cores[1].free #lowest id core first

    c.release(k, now=15.00)
    assert c.cores[0].free
    assert k.finish == 15.00 and k.remaining == 0.00 and k.done


def test_parallel_kernel_uses_mult_cores():
    c = make_cpu(num_cores=4)
    k = make_simple_kernel(0, 100, arrival=0.00)

    time = c.assign(k, now=0.00, num_cores=4)
    assert time == pytest.approx(4.00)
    assert c.free_cores() == []

    c.release(k, now=4.00)
    assert len(c.free_cores()) == 4 #all 4 cores freed by 1 release


def test_cpu_assign_errors():
    c = make_cpu(num_cores=2)
    no_split = make_simple_kernel(0, 100, arrival=0.00)
    no_split.parallelize = False
    with pytest.raises(ValueError):
        c.assign(no_split, now=0.00, num_cores=2) #cant split a kernel that cant be parallelized

    c.assign(make_simple_kernel(1, 100, arrival=0.00), now=0.00)
    c.assign(make_simple_kernel(2, 100, arrival=0.00), now=0.00)
    with pytest.raises(RuntimeError):
        c.assign(make_simple_kernel(3, 100, arrival=0.00), now=0.00) #no free cores left
    with pytest.raises(RuntimeError):
        c.release(make_simple_kernel(4, 100, arrival=0.00), now=0.00) #not running on the cpu


#same case as test_systolic.py: (10 x 40) @ (40 x 50), 32 x 32 array, 1 GHz, 50 GB/s
#loads: 2048 + 512 + 1152 + 288 = 4000 bytes / 50 GB/s = 80 ns
#computes: 4 tiles x 74 cycles = 296 ns (the bytes in each compute need < 74 ns => compute bound)
#total = 80 + 296 = 376 ns
def test_accel_standalone_hand_calc():
    a = make_accel()
    k = make_gmm_kernel(0, 10, 40, 50, arrival=0.00)
    assert a.standalone_ns(k) == pytest.approx(376.00)


def test_accel_one_kernel_at_a_time():
    a = make_accel()
    k0 = make_gmm_kernel(0, 10, 40, 50, arrival=0.00)
    k1 = make_gmm_kernel(1, 10, 40, 50, arrival=0.00)
    k2 = make_gmm_kernel(2, 10, 40, 50, arrival=0.00)

    a.assign(k0, now=0.00)
    assert k0.resource == accelerator and not a.free
    with pytest.raises(RuntimeError):
        a.assign(k1, now=0.00) #busy

    a.enqueue(k1)
    a.enqueue(k2)
    a.release(k0, now=376.00)
    assert a.free and k0.finish == 376.00
    assert a.next_waiting() is k1 #first in first out
    assert a.next_waiting() is k2
    assert a.next_waiting() is None


def test_accel_rejects_bad_kernels():
    a = make_accel()
    with pytest.raises(ValueError):
        a.assign(make_simple_kernel(0, 100, arrival=0.00), now=0.00) #not gemm
    with pytest.raises(ValueError):
        a.enqueue(make_simple_kernel(1, 100, arrival=0.00))
    with pytest.raises(ValueError):
        a.assign(make_gmm_kernel(2, 10, 40, 50, arrival=0.00, dtype_bytes=4), now=0.00) #fp32 on a fp16 array


#plan 1.3 validation: "always core 0" rule wired into the engine
#kernels that find core 0 busy wait in first in first out order
#hand trace (time = n_element / 10):
# id  queue  time  start  finish  wait
#  0     0    10     0      10      0
#  1     2    20    10      30      8
#  2     4    10    30      40     26
#  3    30     5    40      45     10   <= queued at t=30 same time k1 finishes. its queued event was scheduled
#  4    31    10    45      55     14      first (lower id_seq) so it runs first, sees core 0 busy and waits behind k2
def test_always_core0_hand_trace():
    c = make_cpu(num_cores=2)
    engine = Engine()
    waiting = []

    def start(eng, k):
        time = c.assign(k, eng.current_time)
        eng.schedule(eng.current_time + time, finished, k)

    def on_queued(eng, k):
        if c.cores[0].free:
            start(eng, k)
        else:
            waiting.append(k)

    def on_finished(eng, k):
        c.release(k, eng.current_time)
        if waiting:
            start(eng, waiting.pop(0)) #core 0 just freed so it gets picked (lowest id)

    engine.register(queued, on_queued)
    engine.register(finished, on_finished)

    arrivals = [(0.00, 100), (2.00, 200), (4.00, 100), (30.00, 50), (31.00, 100)]
    kernels = [make_simple_kernel(i, n, arrival=t) for i, (t, n) in enumerate(arrivals)]
    for k in kernels:
        engine.schedule(k.queue, queued, k)

    end = engine.main()

    assert [(k.start, k.finish) for k in kernels] == [(0, 10), (10, 30), (30, 40), (40, 45), (45, 55)]
    assert [k.wait for k in kernels] == [0, 8, 26, 10, 14]
    assert end == 55
    assert all(core.free for core in c.cores) #everything released
    assert all(k.done for k in kernels)