"""
this is the cost model for testing the new algorithm: as in
what happens if a kernel starts right now?

it will tackle 2 questions for a candidate, either kernel or resource:
-time: how long the candidate takes, with the kernels that are 
alr running
-harm: how much later the running kernels finish because the 
candidate takes some of their bw. contention_aware only knows the 1st one

time is counted from now. harm is the sum of the extra ns 
over all running kernels, harm_by splits it per running kernel 
so the delay heuristic can weight it by how critical each one is

"""

from dataclasses import dataclass
from sim.kernels import cpu, accelerator
from sim.memory import allocate
from sim.model import cpu_segs, accelerator_segs

eps = 1e-9

@dataclass(frozen=True)
class Estimate:
    time: float #ns from now till the candidate finishes
    harm_by: dict #{running kernel id: extra ns it needs because of the candidate}
    @property
    def harm(self):
        return sum(self.harm_by.values())


class Lane:
    def __init__(self, segments, idx=0, left=1.0):
        self.segments = segments
        self.idx = idx
        self.left = left
        self.skip_done()

    def skip_done(self): #a segment that is alr finished moves on to the next one
        while self.idx < len(self.segments) and self.left <= eps:
            self.idx += 1
            self.left = 1.0

    @property
    def done(self):
        return self.idx >= len(self.segments)

    @property
    def segment(self):
        return self.segments[self.idx]

def segments(kernel, kind, sim, num_cores=1):
    if kind == cpu:
        return cpu_segs(kernel, sim.cpu, num_cores)
    if kind == accelerator:
        return accelerator_segs(kernel, sim.accelerator)
    raise ValueError(f"Unknown resource kind: {kind}")


#the running jobs as lanes, the ones finishing exactly now are left out
def running_lanes(sim):
    lanes = {}
    for kid, job in sim.jobs.items():
        lane = Lane(job.segments, job.idx, job.left)
        if not lane.done:
            lanes[kid] = lane
    return lanes


def play(lanes, memory, degradation):
    lanes = {kid: Lane(l.segments, l.idx, l.left) for kid, l in lanes.items()}
    t, finish = 0.0, {}
    while lanes:
        shares = allocate({kid: l.segment.demand() for kid, l in lanes.items()}, memory, degradation)
        seg_time = {kid: l.segment.duration_ns(shares[kid]) for kid, l in lanes.items()}
        dt = min(l.left * seg_time[kid] for kid, l in lanes.items())
        if dt == float("inf"):
            raise RuntimeError("A kernel gets no bw and can never finish")
        t += dt
        for kid, l in list(lanes.items()):
            l.left = 0.0 if seg_time[kid] == 0 else l.left - dt / seg_time[kid]
            l.skip_done()
            if l.done:
                finish[kid] = t
                del lanes[kid]
    return finish


#starting = [(kernel, kind, num_cores)] already picked in this same pick(), they count as running
def what_if(kernel, kind, sim, num_cores=1, starting=()):
    lanes = running_lanes(sim)
    for k, where, n in starting:
        lanes[k.kernel_id] = Lane(segments(k, where, sim, n or 1))
    memory, degradation = sim.config.memory, sim.degradation
    before = play(lanes, memory, degradation)
    lanes[kernel.kernel_id] = Lane(segments(kernel, kind, sim, num_cores))
    after = play(lanes, memory, degradation)
    harm_by = {kid: after[kid] - before[kid] for kid in before}
    return Estimate(after[kernel.kernel_id], harm_by)

"""
Lane is a throwaway copy of a job’s progress, so the replay never changes the real simulation.
running_lanes reads job.idx and job.left, but not job.bw. When pick() runs, the bandwidth hasn’t been reallocated yet, so job.bw can be out of date. left is always current, because advance_jobs() runs first. A job finishing exactly now has left ≈ 0 on its last segment, so skip_done drops it.
play is the simulator loop in miniature:
Share out the bandwidth.
Find the segment that finishes first (dt).
Move every lane forward by dt.
Record lanes that are done, then repeat.

seg_time == 0 covers zero-length segments, which finish instantly.

starting passes in the kernels already chosen in the same pick(). n or 1 covers the accelerator, where policies pass None for cores.
num_cores is how malleable cores will work later: call what_if(k, cpu, sim, num_cores=n) for each n and compare.

"""