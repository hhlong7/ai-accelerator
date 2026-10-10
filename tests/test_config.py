from dataclasses import replace, FrozenInstanceError

import pytest

from sim.config import CoreConfig, AcceleratorConfig, MemoryConfig, SystemConfig


# defaults should build correctly
def test_defaults_build():
    config = SystemConfig()
    assert config.cores.n_cores > 0
    assert config.accelerator.rows > 0
    assert config.memory.dram_bw_bytes_per_s > 0


# derived values would match hand calculation
def test_derived_values():
    accel = AcceleratorConfig(rows=16, cols=8, clock_hz=2e9)
    assert accel.n_pes == 128
    assert accel.cycle_ns == 0.5

    memory = MemoryConfig(llc_bytes=8 * 1024 * 1024, llc_ways=8)
    assert memory.llc_way_bytes == 1024 * 1024


# swap one part, the rest stays the same
def test_change_one_parameter():
    base = SystemConfig()
    bigger = replace(base, accelerator=replace(base.accelerator, rows=64, cols=64))
    assert bigger.accelerator.n_pes == 64 * 64
    assert bigger.cores == base.cores
    assert bigger.memory == base.memory
    assert base.accelerator.rows == 32 #the original is not changed


# configs are frozen so a run cant change the hardware by accident
def test_config_is_frozen():
    config = SystemConfig()
    with pytest.raises(FrozenInstanceError):
        config.cores.n_cores = 8


# invalid values should raise error
@pytest.mark.parametrize("make", [
    lambda: CoreConfig(n_cores=0),
    lambda: CoreConfig(gflops_per_core=-1.0),
    lambda: CoreConfig(multicore_speedup=1.5),
    lambda: AcceleratorConfig(rows=0),
    lambda: AcceleratorConfig(clock_hz=0),
    lambda: AcceleratorConfig(scratchpad_bytes=-1),
    lambda: MemoryConfig(dram_bw_bytes_per_s=0),
    lambda: MemoryConfig(llc_ways=0),
])
def test_invalid_values_raise(make):
    with pytest.raises(ValueError):
        make()
