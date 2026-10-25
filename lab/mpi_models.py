"""MPI research experiments with exact int64 leaves and explicit limitations.

Cannon/Fox use genuine shifts. Layered SUMMA uses K replication layers.
Recursive and grid-search are transparent CARMA/COSMA-inspired prototypes,
not reproductions of the optimized upstream libraries. Input reading and
output assembly are intentionally included in the reported runtime.
"""

import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from mpi4py import MPI
from exact import read_text, write_text


def part(n, p, r):
    size = n // p + (r < n % p)
    start = r * (n // p) + min(r, n % p)
    return start, start + size


def padded(a, rows, cols):
    out = np.zeros((rows, cols), dtype=np.int64)
    out[: a.shape[0], : a.shape[1]] = a
    return out


def shifted(comm, a, dst, src, tag):
    b = np.empty_like(a)
    comm.Sendrecv(a, dest=dst, sendtag=tag, recvbuf=b, source=src, recvtag=tag)
    return b


def cannon_fox(comm, a, b, fox):
    size = comm.Get_size()
    rank = comm.Get_rank()
    q = math.isqrt(size)
    if q * q != size:
        raise ValueError("Cannon/Fox require a square rank grid")
    m, k = a.shape
    n = b.shape[1]
    mm = (m + q - 1) // q
    kk = (k + q - 1) // q
    nn = (n + q - 1) // q
    aa = padded(a, mm * q, kk * q)
    bb = padded(b, kk * q, nn * q)
    r, c = divmod(rank, q)
    local_a = aa[r * mm : (r + 1) * mm, c * kk : (c + 1) * kk].copy()
    local_b = bb[r * kk : (r + 1) * kk, c * nn : (c + 1) * nn].copy()
    row = comm.Split(r, c)
    out = np.zeros((mm, nn), dtype=np.int64)
    if not fox:
        local_a = shifted(comm, local_a, r * q + (c - r) % q, r * q + (c + r) % q, 1)
        local_b = shifted(
            comm, local_b, ((r - c) % q) * q + c, ((r + c) % q) * q + c, 2
        )
    for step in range(q):
        if fox:
            panel = np.empty_like(local_a)
            if c == (r + step) % q:
                panel[:] = local_a
            row.Bcast(panel, root=(r + step) % q)
        else:
            panel = local_a
        out += panel @ local_b
        if step + 1 < q:
            local_b = shifted(
                comm, local_b, ((r - 1) % q) * q + c, ((r + 1) % q) * q + c, 3
            )
            if not fox:
                local_a = shifted(
                    comm, local_a, r * q + (c - 1) % q, r * q + (c + 1) % q, 4
                )
    row.Free()
    return (
        r * mm,
        min(m, (r + 1) * mm),
        c * nn,
        min(n, (c + 1) * nn),
        out[: max(0, min(mm, m - r * mm)), : max(0, min(nn, n - c * nn))],
    )


def hierarchical_bcast(comm, buf, root):
    """Two-level broadcast, with groups formed in root-relative rank order."""
    rank = comm.Get_rank()
    size = comm.Get_size()
    relative = (rank - root) % size
    group = relative // 2
    local = comm.Split(group, relative)
    leaders = comm.Split(0 if local.Get_rank() == 0 else MPI.UNDEFINED, relative)
    if local.Get_rank() == 0:
        leaders.Bcast(buf, root=0)
        leaders.Free()
    local.Bcast(buf, root=0)
    local.Free()


def summa(comm, a, b, layers, kb, hierarchical=False, grid=None):
    size = comm.Get_size()
    rank = comm.Get_rank()
    if size % layers:
        raise ValueError("layers must divide rank count")
    layer_size = size // layers
    layer = rank // layer_size
    local_rank = rank % layer_size
    pr, pc = grid or MPI.Compute_dims(layer_size, 2)
    if pr * pc != layer_size:
        raise ValueError("invalid grid")
    r, c = divmod(local_rank, pc)
    row = comm.Split(layer * pr + r, c)
    col = comm.Split(layer * pc + c, r)
    fiber = comm.Split(local_rank, layer)
    m, k = a.shape
    n = b.shape[1]
    i0, i1 = part(m, pr, r)
    j0, j1 = part(n, pc, c)
    k0, k1 = part(k, layers, layer)
    out = np.zeros((i1 - i0, j1 - j0), dtype=np.int64)
    for step, start in enumerate(range(k0, k1, kb)):
        end = min(k1, start + kb)
        aa = np.empty((i1 - i0, end - start), np.int64)
        bb = np.empty((end - start, j1 - j0), np.int64)
        ar = step % pc
        br = step % pr
        if c == ar:
            aa[:] = a[i0:i1, start:end]
        if r == br:
            bb[:] = b[start:end, j0:j1]
        if hierarchical:
            hierarchical_bcast(row, aa, ar)
            hierarchical_bcast(col, bb, br)
        else:
            row.Bcast(aa, root=ar)
            col.Bcast(bb, root=br)
        out += aa @ bb
    reduced = np.empty_like(out) if layer == 0 else None
    fiber.Reduce(out, reduced, op=MPI.SUM, root=0)
    row.Free()
    col.Free()
    fiber.Free()
    return (i0, i1, j0, j1, reduced) if layer == 0 else None


def recursive_boxes(shape, ranks, origin=(0, 0, 0)):
    if len(ranks) == 1:
        return [(ranks[0], origin, shape)]
    axis = max(range(3), key=lambda i: shape[i])
    left = len(ranks) // 2
    cut = shape[axis] * left // len(ranks)
    first = list(shape)
    second = list(shape)
    start = list(origin)
    first[axis] = cut
    second[axis] -= cut
    start[axis] += cut
    return recursive_boxes(tuple(first), ranks[:left], origin) + recursive_boxes(
        tuple(second), ranks[left:], tuple(start)
    )


def recursive(comm, a, b):
    rank = comm.Get_rank()
    m, k = a.shape
    n = b.shape[1]
    boxes = recursive_boxes((m, k, n), list(range(comm.Get_size())))
    _, (i, t, j), (mm, kk, nn) = next(box for box in boxes if box[0] == rank)
    # Each rank contributes only its own index box. Root adds overlapping K
    # contributions; no claim of communication-optimal redistribution here.
    return i, i + mm, j, j + nn, a[i : i + mm, t : t + kk] @ b[t : t + kk, j : j + nn]


def choose_grid(m, k, n, p, memory):
    candidates = []
    for layers in range(1, p + 1):
        if p % layers:
            continue
        for pr in range(1, p // layers + 1):
            if (p // layers) % pr:
                continue
            pc = p // layers // pr
            footprint = 8 * (
                math.ceil(m / pr) * math.ceil(k / layers)
                + math.ceil(k / layers) * math.ceil(n / pc)
                + math.ceil(m / pr) * math.ceil(n / pc)
            )
            # Explicit heuristic, not COSMA's optimal scheduling proof.
            traffic = m * k * (pc - 1) + k * n * (pr - 1) + m * n * (layers - 1)
            if footprint <= memory:
                candidates.append((traffic, footprint, layers, pr, pc))
    if not candidates:
        raise ValueError("no grid fits requested local tile budget")
    return min(candidates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument(
        "--algorithm",
        choices=(
            "cannon",
            "fox",
            "layered-summa",
            "hsumma",
            "recursive",
            "grid-search",
        ),
        default="cannon",
    )
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--kb", type=int, default=64)
    p.add_argument("--memory", type=int, default=64 * 1024 * 1024)
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    try:
        if args.kb < 1 or args.layers < 1:
            raise ValueError("positive kb/layers required")
        if Path(args.input).resolve() == Path(args.output).resolve() or (
            Path(args.output).exists() and Path(args.input).samefile(args.output)
        ):
            raise ValueError("input/output must differ")
        start = time.perf_counter()
        aa, bb, cc = read_text(args.input)
        m = len(aa)
        k = len(bb)
        n = len(bb[0])
        bound = k * max(abs(x) for r in aa for x in r) * max(
            abs(x) for r in bb for x in r
        ) + max(abs(x) for r in cc for x in r)
        if bound > np.iinfo(np.int64).max:
            raise ValueError("int64 range proof failed")
        a, b, c = [np.asarray(x, dtype=np.int64) for x in (aa, bb, cc)]
        if args.algorithm in ("cannon", "fox"):
            result = cannon_fox(comm, a, b, args.algorithm == "fox")
        elif args.algorithm == "recursive":
            result = recursive(comm, a, b)
        elif args.algorithm == "grid-search":
            _, _, layers, pr, pc = choose_grid(m, k, n, comm.Get_size(), args.memory)
            result = summa(comm, a, b, layers, args.kb, grid=(pr, pc))
        else:
            result = summa(comm, a, b, args.layers, args.kb, args.algorithm == "hsumma")
        tiles = comm.gather(result, root=0)
        if rank == 0:
            out = np.zeros((m, n), np.int64)
            for tile in tiles:
                if tile is not None:
                    i0, i1, j0, j1, data = tile
                    out[i0:i1, j0:j1] += data
            out += c
            write_text(args.output, out.tolist())
        comm.Barrier()
        elapsed = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "algorithm": args.algorithm,
                        "np": comm.Get_size(),
                        "total_s": elapsed,
                        "prototype": True,
                    }
                )
            )
    except Exception as exc:
        print(f"rank {rank}: {exc}", flush=True)
        comm.Abort(2)


if __name__ == "__main__":
    main()
