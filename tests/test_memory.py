import pytest

from sim.config import AcceleratorConfig, MemoryConfig
from sim.memory import allocate
from sim.systolic import gemm_phases

#peak of 100 bytes per sec so the numbers are easy to check by hand
memory = MemoryConfig(dram_bw_bytes_per_s=100.0)
inf = float("inf")


#one agent alone gets what it asks for, up to the peak
def test_single_agent_gets_standalone():
    assert allocate({"cpu": 40.0}, memory) == {"cpu": 40.0}
    assert allocate({"cpu": 100.0}, memory) == {"cpu": 100.0}
    assert allocate({"cpu": 250.0}, memory) == {"cpu": 100.0}
    assert allocate({"accel": inf}, memory) == {"accel": 100.0}


#total demand 30 + 50 = 80 fits in 100 => no contention
def test_no_contention_when_demand_fits():
    assert allocate({"cpu": 30.0, "accel": 50.0}, memory) == {"cpu": 30.0, "accel": 50.0}


#total demand 60 + 90 = 150 > 100 => both scaled by 100/150
def test_proportional_share_over_peak():
    allocation = allocate({"cpu": 60.0, "accel": 90.0}, memory)
    assert allocation["cpu"] == pytest.approx(40.0)
    assert allocation["accel"] == pytest.approx(60.0)
    assert sum(allocation.values()) == pytest.approx(100.0)


#step 1.4 preview: 2 identical memory bound kernels each get half => each takes 2x standalone
def test_two_memory_bound_agents_split_in_half():
    assert allocate({"k0": inf, "k1": inf}, memory) == {"k0": 50.0, "k1": 50.0}


#inf counts as the peak: 100 + 50 = 150 > 100 => scaled by 100/150
def test_unlimited_demand_counts_as_peak():
    allocation = allocate({"accel": inf, "cpu": 50.0}, memory)
    assert allocation["accel"] == pytest.approx(200 / 3)
    assert allocation["cpu"] == pytest.approx(100 / 3)


#total allocated never goes over the peak
@pytest.mark.parametrize("demands", [
    {"a": 10.0, "b": 20.0, "c": 30.0},
    {"a": 70.0, "b": 80.0, "c": 90.0},
    {"a": inf, "b": inf, "c": 5.0},
    {"a": 1e12, "b": 1.0},
])
def test_total_never_over_peak(demands):
    allocation = allocate(demands, memory)
    assert sum(allocation.values()) <= 100.0 * (1 + 1e-12)
    for agent, demand in demands.items():
        assert allocation[agent] <= demand


#an agent that wants nothing gets nothing and doesnt count as running
def test_zero_demand():
    assert allocate({}, memory) == {}
    assert allocate({"cpu": 0.0}, memory) == {"cpu": 0.0}
    assert allocate({"cpu": 0.0, "accel": inf}, memory) == {"cpu": 0.0, "accel": 100.0}


#co-run loses 20% of the peak: 2 agents share 80 and not 100, alone its still the full 100
def test_degradation_curve():
    def curve(n_agents):
        return 1.0 if n_agents == 1 else 0.8

    assert allocate({"k0": inf, "k1": inf}, memory, curve) == {"k0": 40.0, "k1": 40.0}
    assert allocate({"k0": inf}, memory, curve) == {"k0": 100.0}
    assert allocate({"k0": 30.0, "k1": 40.0}, memory, curve) == {"k0": 30.0, "k1": 40.0} #70 still fits in 80


#the demand of a systolic phase goes straight in: a load phase alone gets the whole peak
def test_systolic_phase_demand():
    accel = AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2)
    load_phase, compute_phase = gemm_phases(10, 40, 50, accel)[:2]
    demands = {"accel": load_phase.demand_bytes_per_s(accel)}
    assert allocate(demands, memory) == {"accel": 100.0}

    #the compute phase wants 640 bytes in 74 ns, way more than 100 bytes per sec => memory bound
    bw = allocate({"accel": compute_phase.demand_bytes_per_s(accel)}, memory)["accel"]
    assert compute_phase.duration_ns(accel, bw) == pytest.approx(640 / 100.0 * 1e9)


def test_invalid_values_raise():
    with pytest.raises(ValueError):
        allocate({"cpu": -1.0}, memory)
    with pytest.raises(ValueError):
        allocate({"cpu": 10.0}, memory, lambda n_agents: 0.0)
    with pytest.raises(ValueError):
        allocate({"cpu": 10.0}, memory, lambda n_agents: 1.5)
