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
        self.by_id = {} #kernel id -> kernel
        self.children = {} #kernel id to the kernel ids that depend on it
        self.deps_left = {} #kernel id to the set of deps that havent finished yet
        self.arrived = set() #kernel that arrive time has passed

    @property
    def now(self):
        return self.engine.current_time

    def submit(self, kernels):
        ids = [k.kernel_id for k in kernels]
        if len(set(ids)) != len(ids) or any(kid in self.by_id for kid in ids):
            raise ValueError(f"Duplicate kernel ids in: {ids}")
        self.by_id.update({k.kernel_id: k for k in kernels})

        for kernel in kernels:
            for dep in kernel.deps:
                if dep not in self.by_id:
                    raise ValueError(f"Kernel: {kernel.kernel_id} depends on unknown kernel: {dep}")
                self.children.setdefault(dep, []).append(kernel.kernel_id)
            self.deps_left[kernel.kernel_id] = {dep for dep in kernel.deps if not self.by_id[dep].done}
        self.check_no_cycles()

        for kernel in kernels:
            self.kernels.append(kernel)
            self.engine.schedule(kernel.queue, queued, kernel)

    def check_no_cycles(self): #check for cycles in the kernel dep graph
        indegree = {kid: len(k.deps) for kid, k in self.by_id.items()}
        free = [kid for kid, n in indegree.items() if n == 0]
        seen = 0
        while free:
            kid = free.pop()
            seen += 1
            for child in self.children.get(kid, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    free.append(child)
        if seen != len(self.by_id):
            raise ValueError("The kernel dependencies have a cycle")

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

    def release(self, kernel):
        kernel.ready = self.now
        self.policy.on_queued(kernel, self)

    def on_queued(self, eng, kernel):
        self.advance_jobs()
        self.arrived.add(kernel.kernel_id)
        if not self.deps_left[kernel.kernel_id]:
            self.release(kernel) #otherwise it gets released when its last dep finishes
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
        for child_id in self.children.get(kernel.kernel_id, []):
            self.deps_left[child_id].discard(kernel.kernel_id)
            if not self.deps_left[child_id] and child_id in self.arrived:
                self.release(self.by_id[child_id])


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
