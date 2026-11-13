"""
workload is where we define the task graph we feed the simulator

types of wl:
-eschs paper: tree, fork join, random layered, fft, and gauuss, each node will get a random kernel

the generator which would generate the knels:
-takes start id so 2 workloads can be send without same id or id clash
-take a seed so every policy can see the same wokrlooad
-returns a list of kernls with deps already fill in
    
"""

import math
import random
from sim.kernels import make_gmm_kernel, make_simple_kernel

#size of the dag nodes
gemm_dims = [2, 4, 8, 16, 32, 64, 128, 256]
simple_sizes = [2**i for i in range(1, 15)] #2,4,8,...,16384

"""
dnn chains
one transformer encoder layer, 10 kernsl in a chain
attention = (heads*seq x d_head) @ (d_head x seq) => (heads*seq x seq)

"""

# def transformer_layer(start_id=0, seq=128, d=768, heads=12, ffn=3072, arrival=0.0):
#     d_head = d // heads
#     i = start_id
#     layer = [
#         make_gmm_kernel(i, seq, d, 3 * d, arrival),
#         make_gmm_kernel(i + 1, heads * seq, d_head, seq, arrival),
#         make_gmm_kernel(i + 2, heads * seq, d, 3 * d, arrival),
#         make_gmm_kernel(i + 3, heads * seq, seq, d_head, arrival),
#         make_gmm_kernel(i + 4, seq, d, d, arrival),
#         make_gmm_kernel(i + 5, seq * d, arrival, ops_per_element=5, n_in=2),
#         make_gmm_kernel(i + 6, seq, d, ffn, arrival),
#         make_simple_kernel(i + 7, seq * ffn, arrival, ops_per_element=8),
#         make_gmm_kernel(i + 8, seq, ffn, d, arrival),
#         make_simple_kernel(i + 9, seq * d, arrival, ops_per_element=5, n_in=2)
#     ]
#     return chain(layer)

def random_kernel(rng, kernel_id, gemm_frac, arrival, deps):
    if rng.random() < gemm_frac:
        m, k, n = (rng.choice(gemm_dims) for _ in range(3))
        return make_gmm_kernel(kernel_id, m, k, n, arrival, deps=deps)
    n_element = rng.choice(simple_sizes)
    return make_simple_kernel(kernel_id, n_element, arrival, n_in=rng.choice([1, 2]), deps=deps)

def build(n_nodes, edges, seed, gemm_frac, start_id, arrival):
    rng = random.Random(seed)
    parents = [[] for _ in range(n_nodes)]
    for parent, child in edges:
        parents[child].append(start_id + parent)
    return [random_kernel(rng, start_id + v, gemm_frac, arrival, parents[v]) for v in range(n_nodes)]

# => out tree: root forks into fanout kids
def tree(levels=4, fanout=2, inward=False, seed=0, gemm_frac=0.5, start_id=0, arrival=0.0):
    edges, frontier, n = [], [0], 1
    for _ in range(levels - 1):
        next = []
        for p in frontier:
            for _ in range(fanout):
                edges.append((p, n))
                next.append(n)
                n += 1
        frontier = next
    if inward:
        edges = [(c, p) for p, c in edges]
    return build(n, edges, seed, gemm_frac, start_id, arrival)

def fork_join(width=4, stages=3, seed=0, gemm_frac=0.5, start_id=0, arrival=0.0):
    edges, join, n = [], 0, 1
    for _ in range(stages):
        branch = list(range(n, n + width))
        n += width
        edges += [(join, b) for b in branch]
        edges += [(b, n) for b in branch]
        join, n = n, n + 1
    return build(n, edges, seed, gemm_frac, start_id, arrival)

def random_dag(n_nodes=20, n_levels=5, edge_prob=0.2, seed=0, gemm_frac=0.5, start_id=0, arrival=0.0):
    if n_nodes < n_levels:
        raise ValueError(f"Number of nodes: {n_nodes} has to be larger than number oflevels: {n_levels}")
    rng = random.Random(seed)
    level_of = list(range(n_levels)) + [rng.randrange(n_levels) for _ in range(n_nodes - n_levels)]
    level_of.sort()
    levels = [[v for v in range(n_nodes) if level_of[v] == l] for l in range(n_levels)]
    edges = set()
    for l in range(1, n_levels):
        for v in levels[l]:
            for u in levels[l - 1]:
                if rng.random() < edge_prob:
                    edges.add((u, v))
    return build(n_nodes, sorted(edges), seed + 1, gemm_frac, start_id, arrival)

#fft task graph
def fft(points=4, seed=0, gemm_frac=0.5, start_id=0, arrival=0.0):
    if points < 2 or points & (points - 1):
        raise ValueError(f"FFT points: {points} has to be a power of 2 and greater than 2")
    edges, n = [], 1
    frontier = [0]
    while len(frontier) < points:
        next = []
        for p in frontier:
            edges += [(p, n), (p, n + 1)]
            next += [n, n + 1]
            n += 2
        frontier = next
    for s in range(int(math.log2(points))):
        level = list(range(n, n + points))
        for i in range(points):
            edges += [(frontier[i], level[i]), (frontier[i ^ (1 << s)], level[i])]
        frontier, n = level, n + points
    return build(n, edges, seed, gemm_frac, start_id, arrival)

def gaussian_elimination(m=5, seed=0, gemm_frac=0.5, start_id=0, arrival=0.0):
    if m < 2:
        raise ValueError(f"m: {m} must be >= 2")
    node, edges = {}, []
    for k in range(1, m):
        for j in range(k, m + 1):
            node[(k, j)] = len(node)
    for k in range(1, m):
        if k > 1:
            edges.append((node[(k - 1, k)], node[(k, k)]))
        for j in range(k + 1, m + 1):
            edges.append((node[(k, k)], node[(k, j)]))
            if k > 1:
                edges.append((node[(k - 1, j)], node[(k, j)]))
    return build(len(node), edges, seed, gemm_frac, start_id, arrival)


