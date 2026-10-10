import pytest

from sim.config import SystemConfig
from sim.kernels import make_simple_kernel
from sim.policies.base import Policy
from sim.simulator import Simulator


#the base policy never starts anything => the simulator has to catch it
def test_base_policy_plugs_into_simulator():
    p = Policy()
    assert p.pick(None) == []

    sim = Simulator(SystemConfig(), p)
    sim.submit([make_simple_kernel(0, 100, arrival=0.00)])
    with pytest.raises(RuntimeError):
        sim.run()