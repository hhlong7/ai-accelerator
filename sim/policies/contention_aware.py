"""
this is contention aware policy. this represents the new idea of this project.
it's a heft greedy, but estimate_ns() knows about memory contention

"""


from sim.kernels import cpu, accelerator
from sim.memory import allocate
from sim.policies.heft_greedy import HeftGreedy

eps = 1e-9 # time left below this = finished


class ContentionAware(HeftGreedy):
    def __init__(self, alpha = 0.25):
        super().__init__()
        self.alpha = alpha # how fast the correction follows. 0 = never, 1 = only last one
        self.correction = {cpu: 1.0, accelerator: 1.0}
        self.predicted = {} # {kernel_id: {kind: predicted time before correction}}


    # running jobs that still have work left. a job finishing at this same time is still in
    # sim.jobs (its event hasnt been processed yet) but it isnt a competitor anymore
    def active(self, sim):
        return {kid: job for kid, job in sim.jobs.items() if job.time_left_ns() > eps}


    def estimate_ns(self, kernel, kind, sim):
        others = [(kid, job.kernel.resource, job.demand()) for kid, job in self.active(sim).items()]
        others += [(k.kernel_id, where, self.segs(k, where, sim)[0].demand()) for k, where, _ in self.picks]

        # the accel runs 1 at a time, so a kernel going there never shares with the
        # one currently on it.

        demands = {kid: demand for kid, where, demand in others if not (kind == accelerator and where == accelerator)}

        time = 0.0
        bw_for = {} # {demand: bandwidth it gets}, segments with the same demand get the same share

        for seg in self.segs(kernel, kind, sim):
            demand = seg.demand()
            if demand not in bw_for:
                demands[kernel.kernel_id] = demand
                bw_for[demand] = allocate(demands, sim.config.memory, sim.degradation)[kernel.kernel_id]
            time += seg.duration_ns(bw_for[demand])

        self.predicted.setdefault(kernel.kernel_id, {})[kind] = time

        return time * self.correction[kind]


    # when a running kernel will be done, with the bandwidth it gets NOW. pick() runs before the
    # simulator reallocates, so job.bw can still be the old share from before the last kernel finished
    def busy_until(self, kernel, sim):
        job = sim.jobs[kernel.kernel_id]
        if job.time_left_ns() <= eps:
            return sim.now

        demands = {kid: j.demand() for kid, j in self.active(sim).items()}
        bw = allocate(demands, sim.config.memory, sim.degradation)[kernel.kernel_id]
        left = job.left * job.segment.duration_ns(bw)
        later = sum(s.duration_ns(bw) for s in job.segments[job.idx + 1:])

        return sim.now + left + later

    # the learning model using mem contention
    def learn(self, kernel, sim):
        kind = kernel.resource
        ratio = (kernel.finish - kernel.start) / self.predicted.pop(kernel.kernel_id)[kind]
        self.correction[kind] += self.alpha * (ratio - self.correction[kind])
        