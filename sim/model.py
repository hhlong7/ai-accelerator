"""
in a soc, the kernel's speed changes while it runs, for when the accelerator loads the weights, the
cpu kernels lose bandwidth and slow down, and when it stops, they would speed up again

so we would tackle that in here through these calm methods:

- split each running kernel into segments, its phases, and each of that would contain 
        -compute_ns (how long it takes if not worry about memory)
        -bytes for how much it moves to and from the dram

- the segment time with bandwidth with formula = max(compute_ns, bytes / bw)
    => bytes or compute_ns of bandwidth to run at full speed

- track the progress as a fraction of the current segment

- whenver the set of running kernels changes:
    - advance every kernels' progress using its old rate
    - ask memory.allocate() for new mem bw shares
    - set new rates, cancel each kernel's old "segment done" event n schedule new one

=> A segment needs 1000 bytes and 100 ns of compute, so it wants 10 GB/s.

t = 0..50, gets 5 GB/s (shared) -> would take 200 ns -> after 50 ns: 25% done, 75% left
t = 50, other kernel finishes -> now gets 20 GB/s -> full speed (100 ns total)
75% of 100 ns = 75 ns left -> finishes at t = 125
"""

from dataclasses import dataclass
from sim.systolic import gemm_phases

@dataclass(frozen=True)
class Segment:
    compute_ns: float   #time if memory is not limiting
    bytes: int #move from and to dram

    #bytes/sec it needs to run at full compute speed
    def demand(self):
        if self.bytes == 0:
            return 0.00 #pure compute, no need for bw

        if self.compute_ns == 0:
            return float("inf")
        return self.bytes / self.compute_ns * 1e9

    def duration_ns(self, bw):
        if self.bytes == 0:
            return self.compute_ns
        if bw <= 0:
            return float("inf")
        return max(self.compute_ns, self.bytes / bw * 1e9)

#cpu kernel = 1 seg, compute time from cores.py n all its bytes
def cpu_segs(kernel, cpu_cores, num_cores=1):
    return [Segment(cpu_cores.standalone_ns(kernel, num_cores), kernel.moved_bytes)]

#accel kernel = the pahses from systolic.py, cycles turned into ns
def accelerator_segs(kernel, accel):
    m, k, n = kernel.dim
    phases = gemm_phases(m, k, n, accel.config)
    return [Segment(p.cycles * accel.config.cycle_ns, p.moved_bytes) for p in phases]


class Job:
    def __init__(self, kernel, segments):
        if not segments:
            raise ValueError(f"The kernel: {kernel.kernel_id} has no segments")
        self.kernel = kernel
        self.segments = segments
        self.idx = 0  #which segment its on
        self.left = 1.0 #fraction of the current segment left (1 = not started, 0 = done)
        self.rate = 0.0 #fraction of the current segment done per ns
        self.bw = 0.0 #bytes per sec it gets right now
        self.event_id = None #its scheduled segment done event, cancelled when the rate changes

    @property
    def segment(self):
        return self.segments[self.idx]

    @property
    def done(self):
        return self.idx >= len(self.segments)

    def demand(self):
        return self.segment.demand()

    #new bandwidth => new rate
    def set_bw(self, bw):
        self.bw = bw
        time = self.segment.duration_ns(bw)
        self.rate = float("inf") if time == 0 else 1.0 / time

    #move progress forward by dt ns with the current rate
    def advance(self, dt):
        if dt <= 0 or self.rate == 0:
            return
        self.left = max(0.0, self.left - self.rate * dt)

    #time to finish the current segment at the current rate
    def time_left_ns(self):
        if self.rate == float("inf"):
            return 0.0
        if self.rate == 0:
            return float("inf")
        return self.left / self.rate

    #estimate for the whole kernel if it keeps the same bandwidth (policies like heft use this)
    def estimate_left_ns(self):
        later = sum(s.duration_ns(self.bw) for s in self.segments[self.idx + 1:])
        return self.time_left_ns() + later

    def next_segment(self):
        self.idx += 1
        self.left = 1.0