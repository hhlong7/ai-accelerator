"""
plans for the new algorithm (step 2b): which resource each kernel goes on
- core_plan: whatever is faster alone (the placeholder, and the ilp=False ablation)
- ilp_plan: one whole-dag ilp
"""

from sim.kernels import cpu, accelerator


#placeholder plan: {kernel id: cpu or accelerator} whatever is faster alone with 1 core
def core_plan(kernels, cost):
    plan = {}
    for k in kernels:
        c = cost[k.kernel_id]
        on_accel = accelerator in c and c[accelerator] < c[cpu]
        plan[k.kernel_id] = accelerator if on_accel else cpu 
    return plan
