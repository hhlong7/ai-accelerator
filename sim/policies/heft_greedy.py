"""
this is heft greedy (StarPU style):

each kernel goes to the resource where it's expected to finish the earliest.
expected finish = when resource is free + how long kernels takes at that resource
the times come from just standalone model --> doesn't know about contention yet,
that's for contention aware

note: qilin style and contention aware are the same policy but with different
estimates for time bc one doesn't know about contention vs knows

a kernel can decide to wait for a busy resource if it determines that it's 
faster there than using a free one! for example, waiting for the accel when 
the cpu would take much longer. just stays in line and looked at again on the 
next pick


"""

from sim.kernels import cpu, accelerator
from sim.model import cpu_segs, accelerator_segs
from sim.policies.base import Policy


class HeftGreedy(Policy):
    def __init__(self):
        self.waiting = [] # empty array, kernels not started
        self.running = [] # kernels that started, not finished
        self.picks = [] # what pick() is starting so far

    def on_queued(self, kernel, sim):
        self.waiting.append(kernel)

    def fits_accel(self, kernel, sim):
        return kernel.can_run_on(accelerator) and kernel.dtype_bytes == sim.config.accelerator.dtype_bytes

    # uses model.py segments
    def segs(self, kernel, kind, sim):
        if kind == cpu:
            return cpu_segs(kernel, sim.cpu, 1)
        return accelerator_segs(kernel, sim.accelerator)

    # calc how long kernel takes on this resource
    def estimate_ns(self, kernel, kind, sim):
        peak = sim.config.memory.dram_bw_bytes_per_s

        return sum(s.duration_ns(peak) for s in self.segs(kernel, kind, sim))


    # calc expected finish
    def busy_until(self, kernel, sim):
        return max(sim.now, kernel.start + self.estimate_ns(kernel, kernel.resource, sim))


    # calls every time a kernel finishes, only policies with history use it
    def learn(self, kernel, sim):
        pass


    def pick(self, sim):
        for kernel in [k for k in self.running if k.done]:
            self.running.remove(kernel)
            self.learn(kernel, sim)

        free_cores = len(sim.cpu.free_cores())
        accel_free = sim.accelerator.free

        # update when each core and accel are expected to be free
        core_at = sorted(sim.now if c.free else self.busy_until(c.current, sim) for c in sim.cpu.cores)
        accel_at = sim.now if accel_free else self.busy_until(sim.accelerator.current, sim)

        self.picks = []
        still = []

        for kernel in self.waiting:
            finish, kind = core_at[0] + self.estimate_ns(kernel, cpu, sim), cpu

            if self.fits_accel(kernel, sim):
                accel_finish = accel_at + self.estimate_ns(kernel, accelerator, sim)

                if accel_finish < finish:
                    finish, kind=  accel_finish, accelerator

            if kind == cpu:
                start_now = free_cores > 0

                if start_now:
                    free_cores -= 1

                core_at[0] = finish
                core_at.sort()
            else:
                start_now = accel_free
                accel_free = False
                accel_at = finish

            if start_now:
                self.picks.append((kernel, kind, 1 if kind == cpu else None))
                self.running.append(kernel)
            else:
                still.append(kernel)

        self.waiting = still
        return self.picks