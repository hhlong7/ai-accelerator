"""
this is for the timing of the systolic array. it runs a mat mul
(m x k) @ (k x n) = (m x n). the weigths (k x n) are cut into
"tiles" bc they dont fit in the rows x cols array all at once. each tile is 2 phases:
each tile is 2 phases:
    - Load: bring tile from mem into array; bandwidth heavy
    - Compute: stream the m input rows thru the array, m + rows + cols cycles



assumptions of this model:
- tiles go n tile by n tile, and for each n tile thru all the k tiles. the partial sums stay on the accel
  so the output of an n tile is written to memory once, on its last k tile
- if the whole input (m x k) fits in the scratchpad its read from memory once and reused for the
  other n tiles, if not its re-fetched for every tile
- the tiles on the edge only move the bytes they use but still take the full m + rows + cols cycles
- load has 0 cycles, its time only depends on the bandwidth it gets

"""

import math
from dataclasses import dataclass

#phase kind
load = "load"
compute = "compute"


@dataclass(frozen=True)
class Phase:
    kind: str
    cycles: int     #min time of the phase when memory is not the limit
    read_bytes: int
    write_bytes: int = 0

    @property
    def moved_bytes(self):
        return self.read_bytes + self.write_bytes

    #bandwidth the phase needs to run at full speed, a phase with 0 cycles takes all it can get
    def demand_bytes_per_s(self, accel):
        if self.cycles == 0:
            return float("inf")
        return self.moved_bytes / (self.cycles * accel.cycle_ns) * 1e9

    #time of the phase with this bandwidth, bounded by compute or by memory whichever is slower
    def duration_ns(self, accel, bw_bytes_per_s):
        if bw_bytes_per_s <= 0:
            raise ValueError(f"bw_bytes_per_s: {bw_bytes_per_s} must be positive")
        compute_ns = self.cycles * accel.cycle_ns
        memory_ns = self.moved_bytes / bw_bytes_per_s * 1e9
        return max(compute_ns, memory_ns)


#number of tiles along k and along n
def tile_counts(k, n, accel):
    return math.ceil(k / accel.rows), math.ceil(n / accel.cols)


#the phases of one general mat mul on the accelerator, in the order they run
def gemm_phases(m, k, n, accel):
    if min(m, k, n) <= 0:
        raise ValueError(f"Dimensions of the general MatMul: {(m, k, n)} must be positive")
    tiles_k, tiles_n = tile_counts(k, n, accel)
    dtype_bytes = accel.dtype_bytes
    input_fits = m * k * dtype_bytes <= accel.scratchpad_bytes
    compute_cycles = m + accel.rows + accel.cols

    tiles = []  #(weight bytes, input bytes, output bytes) per tile
    for j in range(tiles_n):
        n_used = min(accel.cols, n - j * accel.cols)
        for i in range(tiles_k):
            k_used = min(accel.rows, k - i * accel.rows)
            weight_bytes = k_used * n_used * dtype_bytes
            in_bytes = m * k_used * dtype_bytes if (j == 0 or not input_fits) else 0
            out_bytes = m * n_used * dtype_bytes if i == tiles_k - 1 else 0
            tiles.append((weight_bytes, in_bytes, out_bytes))

    phases = []
    if accel.double_buffered:
        #the weights of the next tile load while the current tile computes, only the 1st load is alone
        phases.append(Phase(load, 0, tiles[0][0]))
        for t, (_, in_bytes, out_bytes) in enumerate(tiles):
            next_weight_bytes = tiles[t + 1][0] if t + 1 < len(tiles) else 0
            phases.append(Phase(compute, compute_cycles, in_bytes + next_weight_bytes, out_bytes))
    else:
        for weight_bytes, in_bytes, out_bytes in tiles:
            phases.append(Phase(load, 0, weight_bytes))
            phases.append(Phase(compute, compute_cycles, in_bytes, out_bytes))
    return phases


def total_cycles(phases):
    return sum(p.cycles for p in phases)


def total_moved_bytes(phases):
    return sum(p.moved_bytes for p in phases)
