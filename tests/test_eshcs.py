from pytest import approx
from sim.kernels import make_simple_kernel
from sim.policies.eshcs import (levels, knapsack, allocate_hw, successors, earliest_start,
                                critical_path, latest_finish, successor_rank, urgency, priorities)


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




"""

part b

"""




#part B: diamond 0 -> {1, 2} -> 3, plus 4 that only depends on 0
#execution times 2, 4, 1, 3, 1 => critical path 0 -> 1 -> 3 = 9
ET = {0: 2, 1: 4, 2: 1, 3: 3, 4: 1}


def diamond():
    return [node(0), node(1, [0]), node(2, [0]), node(3, [1, 2]), node(4, [0])]


def test_successors():
    assert successors(diamond()) == {0: [1, 2, 4], 1: [3], 2: [3], 3: [], 4: []}


#3 waits for the slower of its deps: 1 finishes at 2 + 4 = 6
def test_earliest_start_and_critical_path():
    est = earliest_start(diamond(), ET)
    assert est == {0: 0, 1: 2, 2: 2, 3: 6, 4: 2}
    assert critical_path(est, ET) == 9


#a release time later than the deps pushes the start back
def test_earliest_start_waits_for_release():
    ks = [node(0), make_simple_kernel(1, 100, arrival=10.0, deps=[0])]
    assert earliest_start(ks, {0: 2, 1: 1}) == {0: 0, 1: 10.0}


#2 has to finish by 6 so 3 (et 3) can still end at 9. 0 has to finish by 6 - 4 = 2 for kernel 1
def test_latest_finish():
    ks = diamond()
    lft = latest_finish(ks, ET, successors(ks), 9)
    assert lft == {0: 2, 1: 6, 2: 6, 3: 9, 4: 9}


def test_successor_rank():
    assert successor_rank(successors(diamond())) == {0: 1.0, 1: approx(1 / 3), 2: approx(1 / 3), 3: 0.0, 4: 0.0}


#kernel 2: et 1, est 2, lft 6 => window 4 at t = 0, 2 at t = 4, 1 at t = 5 (has to start now)
def test_urgency_grows_as_time_runs_out():
    assert urgency(1, 2, 6, now=0) == approx(0.25)
    assert urgency(1, 2, 6, now=4) == approx(0.5)
    assert urgency(1, 2, 6, now=5) == 1.0
    assert urgency(1, 2, 6, now=8) == 1.0 #past its latest start, still 1


#on the critical path (0, 1, 3) urgency is 1. kernel 1 beats kernel 2, which beats kernel 4
def test_priorities_favor_the_critical_path():
    ks = diamond()
    succ = successors(ks)
    est = earliest_start(ks, ET)
    lft = latest_finish(ks, ET, succ, critical_path(est, ET))
    pri = priorities(ks, ET, est, lft, successor_rank(succ), succ, now=0)
    assert pri[3] == 1.0
    assert pri[4] == approx(1 / 7)
    assert pri[2] == approx(0.25 + 1 / 3)
    assert pri[1] == approx(1 + 1 / 3)
    assert pri[0] == approx(1 + 4 / 3)
    assert pri[1] > pri[2] > pri[4]

