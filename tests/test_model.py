import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig
from sim.cores import CPU, Accelerator
from sim.kernels import make_gmm_kernel, make_simple_kernel
from sim.model import Segment, Job, cpu_segs, accelerator_segs

inf = float("inf")


#10 gflops per core => a simple kernel with n elements (fpop = n) takes n / 10 ns
def make_cpu(num_cores=2):
    config_core = CoreConfig(n_cores=num_cores, gflops_per_core=10.0, multicore_speedup=0.5)
    return CPU(config_core)


def make_accel():
    config_accel = AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2)
    config_mem = MemoryConfig(dram_bw_bytes_per_s=50e9)
    return Accelerator(config_accel, config_mem)


#100 ns of compute, 1000 bytes => needs 1000 bytes / 100 ns = 10 GB/s to run at full speed
def test_segment_demand_and_duration():
    s = Segment(100.0, 1000)
    assert s.demand() == pytest.approx(10e9)
    assert s.duration_ns(20e9) == pytest.approx(100.0) #more bw than it needs => compute bound
    assert s.duration_ns(10e9) == pytest.approx(100.0) #exactly enough
    assert s.duration_ns(5e9) == pytest.approx(200.0)  #half of what it wants => memory bound, 2x slower


def test_segment_edge_cases():
    load = Segment(0.0, 2048)   #pure transfer like a weight load
    assert load.demand() == inf
    assert load.duration_ns(50e9) == pytest.approx(40.96)

    no_mem = Segment(74.0, 0)   #pure compute
    assert no_mem.demand() == 0.0
    assert no_mem.duration_ns(0.0) == pytest.approx(74.0)


#simple kernel 100 elems, 1 input: fpop = 100 => 10 ns on 1 core, bytes = 100 * 2 * (1 + 1) = 400
def test_cpu_segs():
    c = make_cpu()
    k = make_simple_kernel(0, 100, arrival=0.00)
    assert cpu_segs(k, c, num_cores=1) == [Segment(10.0, 400)]
    assert cpu_segs(k, c, num_cores=2)[0].compute_ns == pytest.approx(100 / 15)


#same gemm as test_systolic and test_cores: 4 tiles => 8 segments (load, compute) x 4
def test_accelerator_segs_match_systolic_and_standalone():
    a = make_accel()
    k = make_gmm_kernel(0, 10, 40, 50, arrival=0.00)
    segs = accelerator_segs(k, a)
    assert len(segs) == 8
    assert sum(s.compute_ns for s in segs) == pytest.approx(296.0) #4 x 74 cycles at 1 GHz
    assert sum(s.bytes for s in segs) == 800 + 4000 + 1000
    #alone with the full bandwidth the rate model has to give the same time as cores.py
    assert sum(s.duration_ns(50e9) for s in segs) == pytest.approx(a.standalone_ns(k))


#the worked example in the docstring: segment wants 10 GB/s for 100 ns
#gets 5 GB/s for the 1st 50 ns => 50 / 200 = 25% done, 75% left
#then gets 20 GB/s => full speed, 75% of 100 ns = 75 ns left => done at 125 ns
def test_job_progress_carries_over_when_bw_changes():
    k = make_simple_kernel(0, 100, arrival=0.00)
    job = Job(k, [Segment(100.0, 1000)])

    job.set_bw(5e9)
    assert job.time_left_ns() == pytest.approx(200.0)
    job.advance(50.0)
    assert job.left == pytest.approx(0.75)

    job.set_bw(20e9)
    assert job.time_left_ns() == pytest.approx(75.0)


def test_job_moves_thru_segments():
    k = make_simple_kernel(0, 100, arrival=0.00)
    job = Job(k, [Segment(0.0, 1000), Segment(50.0, 0)]) #load then compute

    assert job.demand() == inf
    job.set_bw(10e9)
    assert job.time_left_ns() == pytest.approx(100.0)       #1000 bytes / 10 GB/s
    assert job.estimate_left_ns() == pytest.approx(150.0)   #+ 50 ns compute after

    job.next_segment()
    assert not job.done and job.left == 1.0 and job.demand() == 0.0
    job.next_segment()
    assert job.done


def test_job_guards():
    k = make_simple_kernel(0, 100, arrival=0.00)
    with pytest.raises(ValueError):
        Job(k, []) #no segments

    instant = Job(k, [Segment(0.0, 0)]) #nothing to do
    instant.set_bw(0.0)
    assert instant.time_left_ns() == 0.0

    stuck = Job(k, [Segment(10.0, 1000)]) #needs memory but gets none
    stuck.set_bw(0.0)
    assert stuck.time_left_ns() == inf
    stuck.advance(5.0) #no crash, no progress
    assert stuck.left == 1.0