"""

this is for the FIFO policy, which is the most simple:

Kernels start in the order they are submitted, and run to completion.
It first tries the accelerator (if the kernel can run there), then the CPU.
If nothing is free, the Kernel just waits, and everyone must wait as well

"""

from sim.kernels import cpu, accelerator
from sim.policies.base import Policy

class Fifo(Policy):
    def __init__(self):
        self.waiting = [] # empty array suggests kernels not started yet

    def on_queued(self, kernel, sim):
        self.waiting.append(kernel) # kernel in queue, waiting


    # tries the accelerator first, then cpu, else just waits
    def pick(self, sim):
        picks = []
        free_cores = len(sim.cpu.free_cores())
        accel_free = sim.accelerator.free

        while self.waiting:
            kernel = self.waiting[0]

            if accel_free and self.fits_accel(kernel, sim):
                picks.append((kernel, accelerator, None))
                accel_free = False
            elif free_cores > 0:
                picks.append((kernel, cpu, 1))
                free_cores -= 1
            else:
                break
            self.waiting.pop(0)

        return picks

    