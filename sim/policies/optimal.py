"""

optimal.py is the benchmark for every other policy (1.8)
it has 2 parts:

1. the best schedule for a small dag, found by brute force: try every way to place the kernels
  and every order to start them in, run each one in the simulator, keep the fastest
2. lower bounds: makespans that no schedule can beat, even on dags too big to brute force

replay policy:
the search needs a policy that does exactly what its told. replay gets a plan, which is:
- order: the kernel ids in the order they start
- place: {kernel_id: (kind, num_cores)}, where each kernel runs

kernels start in exactly that order, each one as early as it can (its deps are done and its
resource is free). if the next kernel cant start yet, nobody behind it starts either, even if
their resource is free, so the order really is the start order.
"""


from sim.kernels import cpu, accelerator
from sim.policies.base import Policy

class Replay(Policy):

    def __init__(self, order, place):
        self.order = list(order)
        self.place = place
        self.next = 0
        self.ready = {}

    def on_queued(self, kernel, sim):
        self.ready[kernel.kernel_id] = kernel

    def pick(self, sim):
        picks = []
        free_cores = len(sim.cpu.free_cores())
        accel_free = sim.accelerator.free

        while self.next < len(self.order):
            kid = self.order[self.next]
            if kid not in self.ready:
                break
            kind, num_cores = self.place[kid]

            if kind == accelerator:
                if not accel_free:
                    break
                accel_free = False
                picks.append((self.ready[kid], accelerator, None))
            else:
                if free_cores < num_cores:
                    break
                free_cores -= num_cores
                picks.append((self.ready[kid], cpu, num_cores))
            self.next += 1

        return picks