import pytest

from sim.config import AcceleratorConfig
from sim.systolic import load, compute, Phase, tile_counts, gemm_phases, total_cycles, total_moved_bytes

#the hand traced case for most tests: (10 x 40) @ (40 x 50) on a 32 x 32 array, 2 bytes per value
#tiles_k = ceil(40/32) = 2, tiles_n = ceil(50/32) = 2 => 4 tiles
#input = 10*40*2 = 800 bytes, weights = 40*50*2 = 4000 bytes, output = 10*50*2 = 1000 bytes
M, K, N = 10, 40, 50


def make_accel(**changes):
    return AcceleratorConfig(rows=32, cols=32, clock_hz=1e9, dtype_bytes=2, **changes)


def test_tile_counts():
    accel = make_accel()
    assert tile_counts(40, 50, accel) == (2, 2)
    assert tile_counts(32, 64, accel) == (1, 2) #exact multiple, no extra tile
    assert tile_counts(33, 65, accel) == (2, 3) #one over => one more tile
    assert tile_counts(1, 1, accel) == (1, 1)   #smaller than the array


#each tile computes for m + rows + cols = 10 + 32 + 32 = 74 cycles, 4 tiles => 296
def test_cycles_match_hand_calc():
    phases = gemm_phases(M, K, N, make_accel())
    assert [p.kind for p in phases] == [load, compute] * 4
    assert [p.cycles for p in phases if p.kind == compute] == [74] * 4
    assert [p.cycles for p in phases if p.kind == load] == [0] * 4
    assert total_cycles(phases) == 296


#weights per tile = k_used * n_used * 2, the edge tiles are smaller: k_used is 32 then 8, n_used is 32 then 18
def test_weight_bytes_per_tile():
    phases = gemm_phases(M, K, N, make_accel())
    assert [p.read_bytes for p in phases if p.kind == load] == [2048, 512, 1152, 288]
    assert sum(p.read_bytes for p in phases if p.kind == load) == 4000


#input fits in the scratchpad => read once, so the traffic is the min: input + weights + output
def test_bytes_when_input_fits():
    phases = gemm_phases(M, K, N, make_accel(scratchpad_bytes=800))
    computes = [p for p in phases if p.kind == compute]
    assert [p.read_bytes for p in computes] == [640, 160, 0, 0]     #input only on the 1st n tile
    assert [p.write_bytes for p in computes] == [0, 640, 0, 360]    #output on the last k tile of each n tile
    assert total_moved_bytes(phases) == 800 + 4000 + 1000


#input doesnt fit => its fetched again for the 2nd n tile, 800 more bytes
def test_bytes_when_input_refetched():
    phases = gemm_phases(M, K, N, make_accel(scratchpad_bytes=799))
    computes = [p for p in phases if p.kind == compute]
    assert [p.read_bytes for p in computes] == [640, 160, 640, 160]
    assert total_moved_bytes(phases) == 1600 + 4000 + 1000


#double buffered: same cycles and bytes, but the next tile's weights load during the compute
def test_double_buffered():
    phases = gemm_phases(M, K, N, make_accel(scratchpad_bytes=800, double_buffered=True))
    assert [p.kind for p in phases] == [load, compute, compute, compute, compute]
    assert phases[0].read_bytes == 2048
    assert [p.read_bytes for p in phases[1:]] == [640 + 512, 160 + 1152, 288, 0]
    assert total_cycles(phases) == 296
    assert total_moved_bytes(phases) == 800 + 4000 + 1000


#74 cycles at 1 GHz = 74 ns of compute, 1000 bytes of traffic
def test_phase_duration_and_demand():
    accel = make_accel()
    phase = Phase(compute, 74, 600, 400)
    assert phase.duration_ns(accel, 50e9) == pytest.approx(74.0)    #memory needs 20 ns => compute bound
    assert phase.duration_ns(accel, 1e9) == pytest.approx(1000.0)   #memory needs 1000 ns => memory bound
    assert phase.demand_bytes_per_s(accel) == pytest.approx(1000 / 74e-9)

    weights = Phase(load, 0, 2048)
    assert weights.duration_ns(accel, 1e9) == pytest.approx(2048.0)
    assert weights.demand_bytes_per_s(accel) == float("inf")


def test_invalid_values_raise():
    accel = make_accel()
    with pytest.raises(ValueError):
        gemm_phases(0, K, N, accel)
    with pytest.raises(ValueError):
        Phase(compute, 74, 1000).duration_ns(accel, 0)
