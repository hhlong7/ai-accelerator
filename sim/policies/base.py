"""
this is the base for every scheduling policy (1.8). the simulator (simulator.py) calls:
- on_queued(kernel, sim): once when a kernel arrives, the policy can store it in its own queues
- pick(sim): after every arrival and every kernel finish, returns what to start RIGHT NOW as a
  list of (kernel, kind, num_cores). kind is cpu or accelerator from kernels.py,
  num_cores is ignored for the accelerator. return [] to start nothing

rules:
- dont pick more than whats free, the simulator starts them in order and raises if a resource is busy
- every kernel has to be started eventually, or sim.run() raises

what the policy can read from sim:
- sim.now: current time in ns
- sim.cpu: CPU from cores.py (free_cores(), standalone_ns(kernel, num_cores))
- sim.accelerator: Accelerator from cores.py (free, standalone_ns(kernel), enqueue(), next_waiting())
- sim.jobs: {kernel_id: Job} of running kernels (job.bw, job.demand(), job.estimate_left_ns())
- sim.config: the SystemConfig (cores, accelerator, memory)
"""


class Policy:
    #called once when a kernel arrives
    def on_queued(self, kernel, sim):
        pass

    #called after every arrival and kernel finish, returns [(kernel, kind, num_cores), ...] to start now
    def pick(self, sim):
        return []