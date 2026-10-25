"""One-pass byte-partitioned text decoding with MPI token/line prefix counts.

Python correctness prototype for P04. Each input byte is scanned by its owner,
with only token-boundary overlap. Matrix values are routed by numeric index.
No root preprocessing pass or binary input cache is used.
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
import mmap
from pathlib import Path
import time
import numpy as np
from mpi4py import MPI
from exact import write_text


def part(n, p, r):
    start = r * (n // p) + min(r, n % p)
    return start, start + n // p + (r < n % p)


def owner(row, n, p):
    q, rem = divmod(n, p)
    cut = (q + 1) * rem
    return row // (q + 1) if row < cut else rem + (row - cut) // q


def decode(path, m, k, n, comm):
    rank, size = comm.Get_rank(), comm.Get_size()
    whitespace = b" \t\r\n"
    with open(path, "rb") as f, mmap.mmap(
        f.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        lo = len(data) * rank // size
        hi = len(data) * (rank + 1) // size
        nl = data[lo:hi].count(b"\n")
        line_base = comm.exscan(nl) or 0
        total_lines = comm.allreduce(nl) + (int(data[-1] != 10))
        if total_lines != 2 * m + k + 3:
            raise ValueError("wrong number of lines")
        cursor = lo
        line = line_base
        tokens = []
        markers = []
        if lo and data[lo - 1] not in whitespace:
            while cursor < len(data) and data[cursor] not in whitespace:
                cursor += 1
        while cursor < hi:
            if data[cursor] in whitespace:
                line += data[cursor] == 10
                cursor += 1
                continue
            start = cursor
            while cursor < len(data) and data[cursor] not in whitespace:
                cursor += 1
            word = data[start:cursor]
            if word.startswith(b"#"):
                markers.append((word.decode("ascii"), line, len(tokens)))
                continue
            body = word[1:] if word.startswith(b"-") else word
            if not body or any(c < 48 or c > 57 for c in body):
                raise ValueError("invalid token")
            value = int(word)
            if not -99 <= value <= 99:
                raise ValueError("this prototype requires [-99,99]")
            tokens.append((line, value))
        prefix = comm.exscan(len(tokens)) or 0
        if comm.allreduce(len(tokens)) != m * k + k * n + m * n:
            raise ValueError("wrong number of values")
        all_markers = sum(
            comm.allgather([(tag, line, prefix + idx) for tag, line, idx in markers]),
            [],
        )
        if all_markers != [
            ("#A", 0, 0),
            ("#B", m + 1, m * k),
            ("#C", m + k + 2, m * k + k * n),
        ]:
            raise ValueError("invalid markers")
    messages = [[] for _ in range(size)]
    for local, (line, value) in enumerate(tokens):
        index = prefix + local
        if index < m * k:
            matrix = 0
            row, col = divmod(index, k)
            expected_line = 1 + row
            dest = owner(row, m, size)
        elif index < m * k + k * n:
            matrix = 1
            row, col = divmod(index - m * k, n)
            expected_line = m + 2 + row
            dest = owner(row, k, size)
        else:
            matrix = 2
            row, col = divmod(index - m * k - k * n, n)
            expected_line = m + k + 3 + row
            dest = owner(row, m, size)
        if line != expected_line:
            raise ValueError("row width mismatch")
        messages[dest].append((matrix, row, col, value))
    i0, i1 = part(m, size, rank)
    t0, t1 = part(k, size, rank)
    a = np.zeros((i1 - i0, k), np.int64)
    b = np.zeros((t1 - t0, n), np.int64)
    c = np.zeros((i1 - i0, n), np.int64)
    for packet in comm.alltoall(messages):
        for matrix, row, col, value in packet:
            if matrix == 0:
                a[row - i0, col] = value
            elif matrix == 1:
                b[row - t0, col] = value
            else:
                c[row - i0, col] = value
    full_b = np.concatenate(comm.allgather(b), axis=0)
    return a, full_b, c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    for key, default in (("m", 1023), ("k", 2047), ("n", 1023)):
        p.add_argument("--" + key, type=int, default=default)
    args = p.parse_args()
    comm = MPI.COMM_WORLD
    try:
        if min(args.m, args.k, args.n) <= 0:
            raise ValueError("positive dimensions required")
        if Path(args.input).resolve() == Path(args.output).resolve() or (
            Path(args.output).exists() and Path(args.input).samefile(args.output)
        ):
            raise ValueError("input/output must differ")
        if args.k * 99 * 99 + 99 > np.iinfo(np.int64).max:
            raise ValueError("int64 bound exceeded")
        start = time.perf_counter()
        a, b, c = decode(args.input, args.m, args.k, args.n, comm)
        parts = comm.gather(a @ b + c, root=0)
        if comm.Get_rank() == 0:
            write_text(args.output, np.concatenate(parts, axis=0).tolist())
        comm.Barrier()
        elapsed = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)
        if comm.Get_rank() == 0:
            print(
                json.dumps(
                    {
                        "algorithm": "fused-parse",
                        "np": comm.Get_size(),
                        "total_s": elapsed,
                        "prototype": True,
                    }
                )
            )
    except Exception as exc:
        print(f"rank {comm.Get_rank()}: {exc}", flush=True)
        comm.Abort(2)


if __name__ == "__main__":
    main()
