from sim.kernels import make_simple_kernel
from sim.policies.eshcs import levels, knapsack, allocate_hw


def node(kernel_id, deps=()):
    return make_simple_kernel(kernel_id, 100, arrival=0.00, deps=deps)


#diamond: 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
def test_levels_diamond():
    ks = [node(0), node(1, [0]), node(2, [0]), node(3, [1, 2])]
    assert levels(ks) == [[0], [1, 2], [3]]


#2 depends on 0 and 1, so it goes after 1 even though 0 -> 2 is a direct edge
def test_levels_skip_edge():
    ks = [node(0), node(1, [0]), node(2, [0, 1])]
    assert levels(ks) == [[0], [1], [2]]


#{1, 2} saves 40 for a cost of 20, {0, 1} only saves 30
def test_knapsack_picks_the_best_total():
    gain = {0: 10, 1: 20, 2: 20}
    cost = {0: 2, 1: 10, 2: 10}
    assert knapsack([0, 1, 2], gain, cost, 20) == {1, 2}
    assert knapsack([0, 1, 2], gain, cost, 0) == set()


#speedups 2, 4, 4, 6 => group sums 2, 8, 6, so group [1, 2] goes first
def test_allocate_whole_groups_biggest_speedup_first():
    groups = [[0], [1, 2], [3]]
    w_sw = {0: 10, 1: 40, 2: 40, 3: 12}
    w_hw = {0: 5, 1: 10, 2: 10, 3: 2}
    assert allocate_hw(groups, w_sw, w_hw, 20) == {1, 2}
    #5 left after [1, 2] => [3] fits too, then [0] (cost 5) doesnt fit in the 3 left
    assert allocate_hw(groups, w_sw, w_hw, 25) == {1, 2, 3}


#group too big => the ilp picks {1, 2} (saves 40), going by speedup would pick {0, 1} (saves 30)
def test_allocate_ilp_when_group_doesnt_fit():
    groups = [[0, 1, 2]]
    w_sw = {0: 12, 1: 30, 2: 30}
    w_hw = {0: 2, 1: 10, 2: 10}
    assert allocate_hw(groups, w_sw, w_hw, 20) == {1, 2}


#ilp takes 0 from the first group (3 left), then the greedy fill takes 2 (cost 2)
def test_allocate_greedy_fill_after_ilp():
    groups = [[0, 1], [2]]
    w_sw = {0: 30, 1: 28, 2: 4}
    w_hw = {0: 10, 1: 10, 2: 2}
    assert allocate_hw(groups, w_sw, w_hw, 13) == {0, 2}


#kernels missing from w_hw (cpu only, or slower on the accelerator) never go there
def test_allocate_skips_kernels_not_allowed():
    groups = [[0, 1]]
    w_sw = {0: 10, 1: 50}
    w_hw = {0: 5}
    assert allocate_hw(groups, w_sw, w_hw, 100) == {0}
