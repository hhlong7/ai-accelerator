"""
this is qilin style policy. the core design is learning from history.
for each resource and kernel type it keeps the kernels that finished there and fits
a line defined by: time = slope * fpop + intercept. then it picks the earliest
finish just like heft. the measured time will include contention.

the line is per kernel type (like the original qilin fits per kernel): memory bound
simple kernels and gemms take very different time per fpop, so 1 line for both
extrapolates badly to big gemms

until a resource has 2 finished kernels of that type with diff sizes, it'll use
the standalone model.

the actual qilin splits one job between cpu and gpu. the kernel we use here just
runs on one resource, so no split


"""


import numpy as np

from sim.kernels import cpu, accelerator
from sim.policies.heft_greedy import HeftGreedy


class QilinStyle(HeftGreedy):
    def __init__(self):
        super().__init__()
        self.history = {} # {(resource, kernel_type): [(fpop, time), ...]}
        self.fit = {} # {(resource, kernel_type): (slope, intercept)}, missing until there's enough history

    def estimate_ns(self, kernel, kind, sim):
        fit = self.fit.get((kind, kernel.kernel_type))
        if fit is None:
            return super().estimate_ns(kernel, kind, sim)
        slope, intercept = fit

        return max(0.0, slope * kernel.fpop + intercept)

    # the learning model using the line, 1 line per resource and kernel type
    def learn(self, kernel, sim):
        key = (kernel.resource, kernel.kernel_type)
        history = self.history.setdefault(key, [])
        history.append((kernel.fpop, kernel.finish - kernel.start))
        sizes = [size for size, _ in history]

        if len(set(sizes)) >= 2:
            slope, intercept = np.polyfit(sizes, [time for _, time in history], 1)
            self.fit[key] = (float(slope), float(intercept))