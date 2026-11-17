"""

optimal.py is the benchmark for every other policy (1.8)
it has 3 parts:

1. the best schedule for a small dag, found by brute force: try every way to place the kernels
  and every order to start them in, run each one in the simulator, keep the fastest
2. lower bounds: makespans that no schedule can beat, even on dags too big to brute force
3. the search: run every start order and every placement through the simulator (with
Replay), keep the one that finishes first.


replay policy:
the search needs a policy that does exactly what its told. replay gets a plan, which is:
- order: the kernel ids in the order they start
- place: {kernel_id: (kind, num_cores)}, where each kernel runs

kernels start in exactly that order, each one as early as it can (its deps are done and its
resource is free). if the next kernel cant start yet, nobody behind it starts either, even if
their resource is free, so the order really is the start order.
"""


from itertools import product

from sim.kernels import cpu, accelerator, fresh, topological
from sim.memory import no_degradation
from sim.simulator import Simulator
from sim.estimate import fits_accel, segments, standalone_ns, fastest_ns
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



'''
part b, lower bounds: no schedule can finish before any of these, on a dag of any size.
the makespan is the time the last kernel finishes, counted from t = 0.

- bytes: every byte has to cross dram, and dram moves at most peak bytes per sec
- critical_path: a chain of deps runs one after another, each kernel at its fastest
(alone, full bandwidth, best resource)
- work: the cpu cores and the accelerator can be busy at the same time, n_cores + 1 of them.
each kernel keeps one busy for at least its fastest 1 core or accelerator time. a kernel on c
cores keeps c cores busy for c * t_c, which is never less than t_1 (c cores are at most c times
faster)
- cpu_work: same idea, but kernels that cant go on the accelerator only have the n_cores cores

'''

def lower_bounds(kernels, config, core_options = (1,)):
    hw = Simulator(config, Policy())
    peak = config.memory.dram_bw_bytes_per_s
    n_cores = config.cores.n_cores
    first = min(k.queue for k in kernels)

    # the fewest bytes each kernel can move,
    # over resources it may use
    def fewest_bytes(k):
        options = [sum(s.bytes for s in segments(k, cpu, hw))]
        if fits_accel(k, hw):
            options.append(sum(s.bytes for s in segments(k, accelerator, hw)))
        return min(options)

    total_bytes = sum(fewest_bytes(k) for k in kernels)

    finish = {} # earliest finish of each kernel
                # with unlimited
    for k in topological(kernels):
        start = max([k.queue] + [finish[d] for d in k.deps])
        finish[k.kernel_id] = start + fastest_ns(k, hw, core_options)

    work = sum(fastest_ns(k, hw) for k in kernels)
    cpu_work = sum(standalone_ns(k, cpu, hw) for k in kernels
                    if not fits_accel(k, hw))

    return {"bytes": first + total_bytes / peak * 1e9,
            "critical_path": max(finish.values()),
            "work": first + work / (n_cores + 1),
            "cpu_work": first + cpu_work / n_cores,}

def lower_bound(kernels, config, core_options = (1,)):
    return max(lower_bounds(kernels, config, core_options).values())



"""
part 3, the search: run every start order and every placement through the simulator (with
Replay), keep the one that finishes first.

- start orders: every order where a kernel comes after all its deps
- placements: each kernel on the accelerator (if it fits) or on the cpu with each core count
in core_options
- it stops early if a schedule reaches the lower bound, nothing can beat that

this is the best LIST schedule: every kernel starts as early as its order lets it. a schedule
that leaves a resource idle on purpose (which can help when memory is contended) is not tried.
the number of schedules grows like n! * 2^n, so its only for small dags (max_kernels)
"""

# every order of the kernel ids where each
# kernel comes after all its deps
def start_orders(kernels):
    deps = {k.kernel_id: set(k.deps) for k in kernels}

    def extend(order, placed):
        if len(order) == len(deps):
            yield list(order)
            return
        for kid in deps:
            if kid not in placed and deps[kid] <= placed:
                order.append(kid)
                placed.add(kid)
                yield from extend(order, placed)
                order.pop()
                placed.remove(kid)

    yield from extend([], set())


# every way to place kernels
def placements(kernels, hw, core_options = (1,)):
    n_cores = hw.config.cores.n_cores
    choices = []
    for k in kernels:
        options = [(cpu, n) for n in core_options if n <= n_cores
                    and (n == 1 or k.parallelize)]
        if fits_accel(k, hw):
            options.append((accelerator, None))
        choices.append(options)
    ids = [k.kernel_id for k in kernels]
    for combo in product(*choices):
        yield dict(zip(ids, combo))

# returns (makespan, order, place) of 
# best scheudle found
def best_schedule(kernels, config, degradation = no_degradation,
                    core_options = (1,), max_kernels = 8):
    if len(kernels) > max_kernels:
        raise ValueError(f"{len(kernels)} kernels is too many, max is {max_kernels}")
    hw = Simulator(config, Policy())
    bound = lower_bound(kernels, config, core_options)
    best = (float("inf"), None, None)

    for place in placements(kernels, hw, core_options):
        for order in start_orders(kernels):
            sim = Simulator(config, Replay(order, place), degradation)
            sim.submit(fresh(kernels))
            end = sim.run()

            if end < best[0]:
                best = (end, order, place)
                if end <= bound * (1 + 1e-9):
                    return best

    return best



    