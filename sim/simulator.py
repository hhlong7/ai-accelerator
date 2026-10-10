"""
this is where it would connect the engine, the cores, and memory model, and the rate model as well as
the policy

every event does these 3 same things:
- advance_job() = bring every running job up to now with its old rate
- change things = start kernels the policy picks, move a job to its next seg, finish a kernel
- reallocate() = new demands, allocate mem, new rates, cancel and reschedule 

events:
- queued: a kernel arrives, the policy hears about it and may start it
- segment_done: a running job finished its current segment => next segment, or the kernel finishes,
  its resource is freed and the policy may start something new

the policy (1.8) needs 2 methods:
- on_queued(kernel, sim): called once when a kernel arrives
- pick(sim): returns a list of (kernel, kind, num_cores) to start right now

"""

from sim.engine import Engine, queued
from sim.cores import CPU, Accelerator
from sim.kernels import cpu, accelerator
from sim.memory import allocate, no_degradation
from sim.model import Job, cpu_segs, accelerator_segs

segment_done = "segment_done"

class Simulator:
    def __init__(self, config, policy, degradation=no_degradation):
        self.config = config
        self.policy = policy
        self.degradation = degradation
        self.engine = Engine()
        self.cpu = CPU(config.cores)
        self.accelerator = Accelerator(config.accelerator, config.memory)
        self.jobs = {}
        self.kernels = [] 
        self.last_update = 0.0 #last time the job progress was advanced
        self.bw_log = [] #(time, total bytes per sec used) every time tge shares change
        self.engine.register(queued, self.on_queued)
        self.engine.register(segment_done, self.on_segment_done)

    @property
    def now(self):
        return self.engine.current_time

    def submit(self, kernels):
        for kernel in kernels:
            self.kernels.append(kernel)
            self.engine.schedule(kernel.queue, queued, kernel)

    #run till everything is done, returns the end time (makespan)
    def run(self, stop=None):
        end = self.engine.main(stop)
        if stop is None:
            stuck = [k.kernel_id for k in self.kernels if not k.done]
            if stuck:
                raise RuntimeError(f"The kernels: {stuck} never finished, the policy never started them")
        return end

    #more kernels arriving at this same time => wait and let the policy see all of them at once
    def more_arrivals_now(self):
        q = self.engine.event_queue
        return bool(q) and q[0][0] == self.now and q[0][2] == queued

    def on_queued(self, eng, kernel):
        self.advance_jobs()
        self.policy.on_queued(kernel, self)
        if not self.more_arrivals_now():
            self.start_picked()
        self.reallocate()

    def on_segment_done(self, eng, job):
        self.advance_jobs()
        job.event_id = None #this event already fired, nothing to cancel
        job.next_segment()
        if job.done:
            self.finish(job)
            self.start_picked() #a resource just got freed
        self.reallocate()

    def start_picked(self):
        for kernel, kind, num_cores in self.policy.pick(self):
            self.start(kernel, kind, num_cores)

    def start(self, kernel, kind, num_cores=1):
        if kernel.kernel_id in self.jobs:
            raise RuntimeError(f"The kernel: {kernel.kernel_id} is already running")
        if kind == cpu:
            self.cpu.assign(kernel, self.now, num_cores)
            segs = cpu_segs(kernel, self.cpu, num_cores)
        elif kind == accelerator:
            self.accelerator.assign(kernel, self.now)
            segs = accelerator_segs(kernel, self.accelerator)
        else:
            raise ValueError(f"Unknown resource kind: {kind}")
        self.jobs[kernel.kernel_id] = Job(kernel, segs)

    def finish(self, job):
        kernel = job.kernel
        if kernel.resource == cpu:
            self.cpu.release(kernel, self.now)
        else:
            self.accelerator.release(kernel, self.now)
        del self.jobs[kernel.kernel_id]


    #step 1: progress everyone up to now with their old rates
    def advance_jobs(self):
        dt = self.now - self.last_update
        for job in self.jobs.values():
            job.advance(dt)
        self.last_update = self.now

    #step 3: new bandwidth shares => new rates => reschedule every segment done event
    def reallocate(self):
        demands = {kid: job.demand() for kid, job in self.jobs.items()}
        shares = allocate(demands, self.config.memory, self.degradation)
        for kid, job in self.jobs.items():
            job.set_bw(shares[kid])
            if job.event_id is not None:
                self.engine.cancel(job.event_id)
            time_left = job.time_left_ns()
            if time_left == float("inf"):
                raise RuntimeError(f"The kernel: {kid} gets no bandwidth and can never finish")
            job.event_id = self.engine.schedule(self.now + time_left, segment_done, job)
        self.bw_log.append((self.now, sum(shares.values())))
