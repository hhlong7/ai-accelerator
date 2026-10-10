"""
this is qilin style policy. the core design is learning from history.
for each resource it keeps of the kernels that finished there and fits
a line defined by: time = slope * fpop + intercept. then it picks the earliest
finish just like heft. the measured time will include contention.

until a resouce has 2 finished kernels of diff sizes and no line, it'll use
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
        self.history = {cpu: [], accelerator: []} # represents the history
        self.fit = {cpu: None, accelerator: None} # line fit (slope, intercept). nothing until there's enough history

    def estimate_ns(self, kernel, kind, sim):
        if self.fit[kind] is None:
            return super().estimate_ns(kernel, kind, sim)
        slope, intercept = self.fit[kind]

        return max(0.0, slope * kernel.fpop + intercept)

    # the learning model using the line
    def learn(self, kernel, sim):
        history = self.history[kernel.resource]
        history.append((kernel.fpop, kernel.finish - kernel.start))
        sizes = [size for size, _ in history]

        if len(set(sizes)) >= 2:
            slope, intercept = np.polyfit(sizes, [time for _, time in history], 1)
            self.fit[kernel.resource] = (float(slope), float(intercept))