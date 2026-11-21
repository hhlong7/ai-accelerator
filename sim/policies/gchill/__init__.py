"""
gchill: the new contention-aware scheduling algorithm
- ilp.py: the policy (ContentionILP), the entry point
- costs.py: (2a) time of each kernel on each resource, alone or with dram sharing
- plan.py: (2b) which resource each kernel goes on
- priority.py: (1) one score per kernel, higher goes first
"""

from sim.policies.gchill.ilp import ContentionILP
