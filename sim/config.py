"""
this is to set configs for CPU, GPU, DRAM & LLC, 
and the entire wrapper object

"""

from dataclasses import dataclass, field


@dataclass(frozen=True)

# config for CPU cores
class CoreConfig:
    # all of the values here are just placeholders!
    
    n_cores: int = 4 # num of cores in CPU
    gflops_per_core: float = 10.0   # speed of the core in giga flops/s
    multicore_speedup: float = 0.9  # each extra core adds this fraction of a core, represents parallel kernels

    # guidelines to make sure all the values are valid (js can't be negative)
    def __post_init__(self):
        if self.n_cores <= 0:
            raise ValueError(f"n_cores: {self.n_cores} must be positive")
        if self.gflops_per_core <= 0:
            raise ValueError(f"gflops_per_core: {self.gflops_per_core} must be positive")
        if not 0 < self.multicore_speedup <= 1:
            raise ValueError(f"multicore_speedup: {self.multicore_speedup} must be in (0, 1]")

# config for GPU core
@dataclass(frozen=True)
class AcceleratorConfig:
    # all of the values here are just placeholders!

    rows: int = 32  #systolic array is rows x cols
    cols: int = 32
    clock_hz: float = 1e9   # frequency of the clock
    scratchpad_bytes: int = 256 * 1024  #on-accel buffer for input activations
    dtype_bytes: int = 2    #bf16, same as kernels.py
    double_buffered: bool = False   #overlap weight load with compute (step 1.5 option)

    # guidelines 
    def __post_init__(self):
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError(f"Array size {(self.rows, self.cols)} must be positive")
        if self.clock_hz <= 0:
            raise ValueError(f"clock_hz: {self.clock_hz} must be positive")
        if self.scratchpad_bytes < 0:
            raise ValueError(f"scratchpad_bytes: {self.scratchpad_bytes} must be >= 0")
        if self.dtype_bytes <= 0:
            raise ValueError(f"dtype_bytes: {self.dtype_bytes} must be positive")

    @property
    def n_pes(self):
        return self.rows * self.cols

    @property
    def cycle_ns(self):
        return 1e9 / self.clock_hz  #length of one clock cycle in ns


@dataclass(frozen=True)

# config for DRAM and LLC
class MemoryConfig:
    dram_bw_bytes_per_s: float = 50e9   # this is the peak DRAM bandwidth, shared by cpu and accel
    llc_bytes: int = 12 * 1024 * 1024   # LLC cache size
    llc_ways: int = 12  # num of ways. ways are the splits between cpu and accel


    # guidelines
    def __post_init__(self):
        if self.dram_bw_bytes_per_s <= 0:
            raise ValueError(f"dram_bw_bytes_per_s: {self.dram_bw_bytes_per_s} must be positive")
        if self.llc_bytes <= 0 or self.llc_ways <= 0:
            raise ValueError(f"LLC size {self.llc_bytes} and ways {self.llc_ways} must be positive")


    @property
    def llc_way_bytes(self):
        return self.llc_bytes // self.llc_ways


# the whole system wrapped up in one config object
# this is what gets passed to simulator
@dataclass(frozen=True)
class SystemConfig:
    cores: CoreConfig = field(default_factory=CoreConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
