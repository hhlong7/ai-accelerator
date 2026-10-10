"""
this is where we have the cores setup (cpu and accelerator) for the kernels to run
these cores would be able to determine if:
- whether if the kernel can run on it
- how long does it take if it is running alone w no memory contention
- assign and release it, filling in the kernel runtie fields of resource, start, finish, and remains
this part is standalone time, rate model would make it into time under contention

"""

from collections import deque
from sim.kernels import cpu, accelerator
from sim.systolic import gemm_phases


class Core:
    def __init__(self, id_core):
        self.id_core = id_core
        self.current = None     #indicate if kernel is running on it, none if idle

    @property
    def free(self):
        return self.current is None

class CPU:
    def __init__(self, config_core):
        self.config = config_core
        self.cores = [Core(i) for i in range(config_core.n_cores)]

    def free_cores(self):
        return [c for c in self.cores if c.free]

    #speedup with num of cores = 1 + (n - 1) * multicore_speedup, like 0.9 => 4 cores = 3.7x not 4x
    def speedup(self, num_cores):
        return 1 + (num_cores - 1) * self.config.multicore_speedup

    #time = fpop / (billions of fpops per sec) in nano sec
    def standalone_ns(self, kernel, num_cores=1):
        return kernel.fpop / (self.config.gflops_per_core * self.speedup(num_cores))

    #assign takes the lowest id free cores, and returns the standalone time in ns
    def assign(self, kernel, now, num_cores=1):
        if not kernel.can_run_on(cpu):
            raise ValueError(f"The kernel: {kernel.kernel_id} cant run on the cpu")
        
        if num_cores < 1:
            raise ValueError(f"Number of Cores: {num_cores} must be at least 1")
        
        if num_cores > 1 and not kernel.parallelize:
            raise ValueError(f"The kernel {kernel.kernel_id} cant be split across {num_cores} cores")
        
        free = self.free_cores()
        if len(free) < num_cores:
            raise RuntimeError(f"The kernel: {kernel.kernel_id} needs {num_cores} free cores, only {len(free)} is free")

        for core in free[:num_cores]:
            core.current = kernel
        kernel.resource = cpu
        kernel.start = now
        return self.standalone_ns(kernel, num_cores)


    def release(self, kernel, now):
        held = [c for c in self.cores if c.current is kernel]
        if not held:
            raise RuntimeError(f"kernel {kernel.kernel_id} is not running on the cpu")
        for core in held:
            core.current = None
        kernel.finish = now
        kernel.remaining = 0.0


class Accelerator:
    def __init__(self, config_accel, config_mem):
        self.config = config_accel
        self.config_mem = config_mem
        self.current = None     #only 1 kernel can run at a time, none if idle
        self.waiting = deque()  #kernels waiting for the accelerator, first in first out

    @property
    def free(self):
        return self.current is None

    #only gemm can run, and the kernel dtype has to match the array or the bytes from systolic.py are wrong
    def check_kernel(self, kernel):
        if not kernel.can_run_on(accelerator):
            raise ValueError(f"The kernel: {kernel.kernel_id} cant run on the accelerator")
        if kernel.dtype_bytes != self.config.dtype_bytes:
            raise ValueError(f"The kernel: {kernel.kernel_id} has dtype_bytes {kernel.dtype_bytes} but the accelerator uses {self.config.dtype_bytes}")

    #time = sum of the phases from systolic.py with the full dram bandwidth in nano sec, no memory contention
    def standalone_ns(self, kernel):
        m, k, n = kernel.dim
        phases = gemm_phases(m, k, n, self.config)
        bw = self.config_mem.dram_bw_bytes_per_s
        return sum(p.duration_ns(self.config, bw) for p in phases)

    #add the kernel to the end of the waiting line
    def enqueue(self, kernel):
        self.check_kernel(kernel)
        self.waiting.append(kernel)

    #next kernel in line, none if nobody is waiting
    def next_waiting(self):
        if self.waiting:
            return self.waiting.popleft()
        return None

    #assign the kernel to the accelerator, and returns the standalone time in ns
    def assign(self, kernel, now):
        self.check_kernel(kernel)
        if not self.free:
            raise RuntimeError(f"The accelerator is busy with kernel: {self.current.kernel_id}")
        self.current = kernel
        kernel.resource = accelerator
        kernel.start = now
        return self.standalone_ns(kernel)

    def release(self, kernel, now):
        if self.current is not kernel:
            raise RuntimeError(f"kernel {kernel.kernel_id} is not running on the accelerator")
        self.current = None
        kernel.finish = now
        kernel.remaining = 0.0
