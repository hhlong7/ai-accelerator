"""
this is contention aware policy. this represents the new idea of this project.
it's a heft greedy, but estimate_ns() knows about memory contention

"""


from sim.kernels import cpu, accelerator
from sim.memory import allocate
from sim.policies.heft_greedy import HeftGreedy


class ContentionAware(HeftGreedy):
    def __init__(self, alpha = 0.25):
        super().__init__()
        self.alpha = alpha # how fast the correction follows. 0 = never, 1 = only last one
        self.correction = {cpu: 1.0, accelerator: 1.0}
        self.predicted = {} # {kernel_id: {kind: predicted time before correction}}


    def estimate_ns(self, kernel, kind, sim):
        others = [(kid, job.kernel.resource, job.demand()) for kid, job in sim.jobs.items()]
        others += [(k.kernel_id, where, self.segs(k, where, sim)[0].demand()) for k, where, _ in self.picks]

        # the accel runs 1 at a time, so a kernel going there never shares with the
        # one currently on it.

        demands = {kid: demand for kid, where, demand in others if not (kind == accelerator and where == accelerator)}

        time = 0.0

        for seg in self.segs(kernel, kind, sim):
            demands[kernel.kernel_id] = seg.demand()
            bw = allocate(demands, sim.config.memory, sim.degradation)[kernel.kernel_id]
            time += seg.duration_ns(bw)

        self.predicted.setdefault(kernel.kernel_id, {})[kind] = time

        return time * self.correction[kind]


    def busy_until(self, kernel, sim):
        return sim.now + sim.jobs[kernel.kernel_id].estimate_left_ns()

    # the learning model using mem contention
    def learn(self, kernel, sim):
        kind = kernel.resource
        ratio = (kernel.finish - kernel.start) / self.predicted.pop(kernel.kernel_id)[kind]
        self.correction[kind] += self.alpha * (ratio - self.correction[kind])
        