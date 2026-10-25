"""Executable exact-arithmetic research models. No floating tolerances.

These Python implementations are correctness/algorithm prototypes, not CPU
speed claims. They use arbitrary precision integers and bounded recursion.
"""

from __future__ import annotations
import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path


def zeros(m, n):
    return [[0] * n for _ in range(m)]


def shape(a):
    if not a or not a[0] or any(len(r) != len(a[0]) for r in a):
        raise ValueError("nonempty rectangular matrix required")
    return len(a), len(a[0])


def add(a, b, sign=1):
    return [[x + sign * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def naive(a, b):
    m, k = shape(a)
    kb, n = shape(b)
    if k != kb:
        raise ValueError("incompatible shapes")
    c = zeros(m, n)
    for i, row in enumerate(a):
        for t, x in enumerate(row):
            for j, y in enumerate(b[t]):
                c[i][j] += x * y
    return c


def blocks(a):
    m, n = shape(a)
    h, w = (m + 1) // 2, (n + 1) // 2
    out = [zeros(h, w) for _ in range(4)]
    for i in range(m):
        for j in range(n):
            out[(i // h) * 2 + j // w][i % h][j % w] = a[i][j]
    return out


def join(q, m, n):
    h, w = shape(q[0])
    return [
        [q[(i // h) * 2 + j // w][i % h][j % w] for j in range(n)] for i in range(m)
    ]


def strassen(a, b, depth=1, leaf=16, winograd=False):
    m, k = shape(a)
    kb, n = shape(b)
    if k != kb:
        raise ValueError("incompatible shapes")
    if depth == 0 or min(m, k, n) <= leaf:
        return naive(a, b)
    a11, a12, a21, a22 = blocks(a)
    b11, b12, b21, b22 = blocks(b)

    def mul(x, y):
        return strassen(x, y, depth - 1, leaf, winograd)

    if not winograd:
        p1 = mul(add(a11, a22), add(b11, b22))
        p2 = mul(add(a21, a22), b11)
        p3 = mul(a11, add(b12, b22, -1))
        p4 = mul(a22, add(b21, b11, -1))
        p5 = mul(add(a11, a12), b22)
        p6 = mul(add(a21, a11, -1), add(b11, b12))
        p7 = mul(add(a12, a22, -1), add(b21, b22))
        q = [
            add(add(add(p1, p4), p5, -1), p7),
            add(p3, p5),
            add(p2, p4),
            add(add(add(p1, p2, -1), p3), p6),
        ]
    else:
        s1 = add(a21, a22)
        s2 = add(s1, a11, -1)
        s3 = add(a11, a21, -1)
        s4 = add(a12, s2, -1)
        t1 = add(b12, b11, -1)
        t2 = add(b22, t1, -1)
        t3 = add(b22, b12, -1)
        t4 = add(t2, b21, -1)
        # Consume each product immediately. Keep only one temporary P matrix.
        q = [zeros((m + 1) // 2, (n + 1) // 2) for _ in range(4)]
        jobs = [
            (a11, b11, (1, 1, 1, 1)),
            (a12, b21, (1, 0, 0, 0)),
            (s4, b22, (0, 1, 0, 0)),
            (a22, t4, (0, 0, -1, 0)),
            (s1, t1, (0, 1, 0, 1)),
            (s2, t2, (0, 1, 1, 1)),
            (s3, t3, (0, 0, 1, 1)),
        ]
        for left, right, coefficients in jobs:
            product = mul(left, right)
            for index, coefficient in enumerate(coefficients):
                if coefficient:
                    q[index] = add(q[index], product, coefficient)
            del product
    return join(q, m, n)


def bitserial(a, b):
    m, k = shape(a)
    _, n = shape(b)
    if any(x < -128 or x > 127 for row in a + b for x in row):
        raise ValueError("bitserial requires signed int8 inputs")
    ap = [
        [sum(((a[i][t] >> bit) & 1) << t for t in range(k)) for bit in range(8)]
        for i in range(m)
    ]
    bp = [
        [sum(((b[t][j] >> bit) & 1) << t for t in range(k)) for bit in range(8)]
        for j in range(n)
    ]
    weights = [1 << i for i in range(7)] + [-128]
    return [
        [
            sum(
                (ap[i][x] & bp[j][y]).bit_count() * weights[x] * weights[y]
                for x in range(8)
                for y in range(8)
            )
            for j in range(n)
        ]
        for i in range(m)
    ]


def four_russians(a, b, group=4):
    m, k = shape(a)
    _, n = shape(b)
    if not 1 <= group <= 8 or any(x < -128 or x > 127 for row in a for x in row):
        raise ValueError("int8 A and group in 1..8 required")
    out = zeros(m, n)
    for t0 in range(0, k, group):
        count = min(group, k - t0)
        table = [None] * (1 << count)
        table[0] = [0] * n
        for mask in range(1, 1 << count):
            bit = (mask & -mask).bit_length() - 1
            table[mask] = [x + y for x, y in zip(table[mask & (mask - 1)], b[t0 + bit])]
        for i in range(m):
            for bit in range(8):
                mask = sum(((a[i][t0 + t] >> bit) & 1) << t for t in range(count))
                weight = (1 << bit) if bit != 7 else -128
                for j in range(n):
                    out[i][j] += weight * table[mask][j]
    return out


def primes():
    n = 101
    while True:
        if all(n % d for d in range(2, math.isqrt(n) + 1)):
            yield n
        n += 1


def crt(a, b):
    m, k = shape(a)
    _, n = shape(b)
    bound = k * max(abs(x) for r in a for x in r) * max(abs(x) for r in b for x in r)
    out, modulus = zeros(m, n), 1
    for p in primes():
        product = naive(
            [[x % p for x in r] for r in a], [[x % p for x in r] for r in b]
        )
        inv = pow(modulus, -1, p)
        for i in range(m):
            for j in range(n):
                out[i][j] += modulus * (((product[i][j] - out[i][j]) * inv) % p)
        modulus *= p
        if modulus > 2 * bound:
            return [[x if x <= modulus // 2 else x - modulus for x in r] for r in out]


def kronecker(a, b):
    m, k = shape(a)
    _, n = shape(b)
    aa = max(abs(x) for row in a for x in row)
    bb = max(abs(x) for row in b for x in row)
    bits = max(1, (4 * k * aa * bb).bit_length())
    mask = (1 << bits) - 1
    left = [sum((x + aa) << (bits * t) for t, x in enumerate(row)) for row in a]
    right = [
        sum((b[t][j] + bb) << (bits * (k - 1 - t)) for t in range(k)) for j in range(n)
    ]
    sums_a, sums_b = list(map(sum, a)), [
        sum(b[t][j] for t in range(k)) for j in range(n)
    ]
    return [
        [
            ((left[i] * right[j] >> (bits * (k - 1))) & mask)
            - bb * sums_a[i]
            - aa * sums_b[j]
            - k * aa * bb
            for j in range(n)
        ]
        for i in range(m)
    ]


def structured(a, b):
    """Exact row deduplication and sparse residual, without input assumptions."""
    m, k = shape(a)
    _, n = shape(b)
    small = [[max(-7, min(7, x)) for x in row] for row in b]
    residual = [
        [(j, x - small[t][j]) for j, x in enumerate(row) if x != small[t][j]]
        for t, row in enumerate(b)
    ]
    cache, out = {}, []
    for row in a:
        key = tuple(row)
        if key not in cache:
            result = naive([row], small)[0]
            for t, x in enumerate(row):
                if x:
                    for j, y in residual[t]:
                        result[j] += x * y
            cache[key] = result
        out.append(cache[key].copy())
    return out


def strassen_scheme():
    return {
        "dims": [2, 2, 2],
        "U": [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ],
        "V": [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        "W": [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0],
            [1, -1, 1, 0, 0, 1, 0],
        ],
    }


def certify(scheme):
    m, k, n = scheme["dims"]
    u, v, w = [
        list(map(lambda r: list(map(Fraction, r)), scheme[key]))
        for key in ("U", "V", "W")
    ]
    rank = len(u)
    if (
        not 1 <= m * k * n <= 512
        or len(v) != rank
        or len(w) != m * n
        or any(len(r) != m * k for r in u)
        or any(len(r) != k * n for r in v)
        or any(len(r) != rank for r in w)
    ):
        raise ValueError("invalid or excessive tensor scheme dimensions")
    for i in range(m * n):
        for x in range(m * k):
            for y in range(k * n):
                expected = int(x // k == i // n and y % n == i % n and x % k == y // n)
                if sum(w[i][r] * u[r][x] * v[r][y] for r in range(rank)) != expected:
                    raise ValueError(f"invalid bilinear certificate at {(i,x,y)}")
    return u, v, w


def inverse(matrix):
    n = len(matrix)
    a = [
        [Fraction(x) for x in row] + [Fraction(i == j) for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col]), None)
        if pivot is None:
            raise ValueError("singular basis")
        a[col], a[pivot] = a[pivot], a[col]
        z = a[col][col]
        a[col] = [v / z for v in a[col]]
        for row in range(n):
            if row != col:
                z = a[row][col]
                a[row] = [x - z * y for x, y in zip(a[row], a[col])]
    return [row[n:] for row in a]


def tensor(a, b, scheme=None, basis=False):
    scheme = scheme or strassen_scheme()
    u, v, w = certify(scheme)
    bm, bk, bn = scheme["dims"]
    m, k = shape(a)
    _, n = shape(b)
    mm, kk, nn = (m + bm - 1) // bm, (k + bk - 1) // bk, (n + bn - 1) // bn

    def partition(x, r, c, h, width):
        parts = [zeros(h, width) for _ in range(r * c)]
        for i, row in enumerate(x):
            for j, val in enumerate(row):
                parts[(i // h) * c + j // width][i % h][j % width] = val
        return parts

    ap, bp = partition(a, bm, bk, mm, kk), partition(b, bk, bn, kk, nn)

    def linear(coeff, parts, cache):
        key = tuple(coeff)
        if key not in cache:
            h, width = shape(parts[0])
            out = zeros(h, width)
            for q, block in zip(coeff, parts):
                if q:
                    for i in range(h):
                        for j in range(width):
                            out[i][j] += q * block[i][j]
            cache[key] = out
        return cache[key]

    if basis:
        # Explicit unimodular input bases. This is a verified alternative-basis
        # prototype, not a claim to reproduce a paper's optimized coefficient set.
        def transform(parts, coefficients):
            count = len(parts)
            p = [[int(i == j) for j in range(count)] for i in range(count)]
            if count > 1:
                p[1][0] = 1
            return [linear(r, parts, {}) for r in p], naive(coefficients, inverse(p))

        ap, u = transform(ap, u)
        bp, v = transform(bp, v)

    def cse(coefficients, parts):
        # Greedy exact two-term CSE. Expanded forms remain algebraically equal.
        from collections import Counter

        rows = [list(row) for row in coefficients]
        while True:
            occurrences = Counter()
            for row in rows:
                terms = [(i, x) for i, x in enumerate(row) if x]
                for i, left in enumerate(terms):
                    for right in terms[i + 1 :]:
                        occurrences[left, right] += 1
            choices = [
                (count, pair) for pair, count in occurrences.items() if count > 1
            ]
            if not choices:
                break
            _, pair = max(choices, key=lambda x: x[0])
            (i, ci), (j, cj) = pair
            coeff = [0] * len(parts)
            coeff[i] = ci
            coeff[j] = cj
            parts.append(linear(coeff, parts, {}))
            for row in rows:
                match = row[i] == ci and row[j] == cj
                if match:
                    row[i] = row[j] = 0
                row.append(int(match))
        return rows, parts

    u, ap = cse(u, ap)
    v, bp = cse(v, bp)
    result = [zeros(mm, nn) for _ in range(bm * bn)]
    # Cache equal linear forms, and consume each product immediately to avoid
    # retaining all rank products. No exponential optimal-schedule claim.
    from collections import Counter

    uses_a, uses_b = Counter(map(tuple, u)), Counter(map(tuple, v))
    ac, bc = {}, {}
    for r in range(len(u)):
        product = naive(linear(u[r], ap, ac), linear(v[r], bp, bc))
        for q, coeff in enumerate(row[r] for row in w):
            if coeff:
                for i in range(mm):
                    for j in range(nn):
                        result[q][i][j] += coeff * product[i][j]
        for coefficients, uses, cache in ((u[r], uses_a, ac), (v[r], uses_b, bc)):
            key = tuple(coefficients)
            uses[key] -= 1
            if uses[key] == 0:
                del cache[key]
        del product
    out = [
        [result[(i // mm) * bn + j // nn][i % mm][j % nn] for j in range(n)]
        for i in range(m)
    ]
    if any(Fraction(x).denominator != 1 for row in out for x in row):
        raise ArithmeticError("nonintegral result")
    return [[int(x) for x in row] for row in out]


from transforms import ntt_gemm, finite_field_fp64, morton

MODELS = {
    "naive": naive,
    "strassen": lambda a, b: strassen(a, b, 2, 1),
    "strassen-winograd": lambda a, b: strassen(a, b, 2, 1, True),
    "alternative-basis": lambda a, b: tensor(a, b, basis=True),
    "tensor": tensor,
    "bitserial": bitserial,
    "four-russians": four_russians,
    "crt": crt,
    "structured": structured,
    "kronecker": kronecker,
    "ntt": ntt_gemm,
    "finite-field-fp64": finite_field_fp64,
    "morton": morton,
}


def read_text(path):
    matrices, current = {}, None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            if line not in ("#A", "#B", "#C") or line in matrices:
                raise ValueError("invalid/duplicate marker")
            current = line
            matrices[current] = []
        else:
            if current is None or not line.strip():
                raise ValueError("invalid row")
            matrices[current].append([int(x) for x in line.split()])
    if list(matrices) != ["#A", "#B", "#C"]:
        raise ValueError("expected #A, #B, #C")
    a, b, c = [matrices[x] for x in ("#A", "#B", "#C")]
    m, k = shape(a)
    kb, n = shape(b)
    if k != kb or shape(c) != (m, n):
        raise ValueError("incompatible dimensions")
    return a, b, c


def write_text(path, d):
    Path(path).write_text("#D\n" + "".join(" ".join(map(str, row)) + "\n" for row in d))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", choices=MODELS, default="strassen")
    p.add_argument("--scheme", type=Path)
    args = p.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve() or (
        Path(args.output).exists() and Path(args.input).samefile(args.output)
    ):
        p.error("input/output must differ")
    start = time.perf_counter()
    a, b, c = read_text(args.input)
    d = (
        tensor(a, b, json.loads(args.scheme.read_text()))
        if args.scheme
        else MODELS[args.model](a, b)
    )
    write_text(args.output, add(d, c))
    print(
        json.dumps(
            {"model": args.model, "total_s": time.perf_counter() - start, "exact": True}
        )
    )


if __name__ == "__main__":
    main()
