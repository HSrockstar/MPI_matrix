"""Exact transform/finite-field prototypes; conversion is included in callers' timing."""

import math


def ntt(values, invert, prime, root):
    a = list(values)
    n = len(a)
    if not n or n & (n - 1) or (prime - 1) % n:
        raise ValueError("NTT length must be a power of two dividing p-1")
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    width = 2
    while width <= n:
        step = pow(root, (prime - 1) // width, prime)
        if invert:
            step = pow(step, -1, prime)
        for start in range(0, n, width):
            w = 1
            for offset in range(width // 2):
                x = a[start + offset]
                y = a[start + offset + width // 2] * w % prime
                a[start + offset] = (x + y) % prime
                a[start + offset + width // 2] = (x - y) % prime
                w = w * step % prime
        width *= 2
    if invert:
        inv = pow(n, -1, prime)
        a = [x * inv % prime for x in a]
    return a


def ntt_gemm(a, b):
    m, k, n = len(a), len(a[0]), len(b[0])
    bound = k * max(abs(x) for r in a for x in r) * max(abs(x) for r in b for x in r)
    size = 1 << ((2 * k - 2).bit_length())
    available = ((998244353, 3), (1004535809, 3), (469762049, 3))
    selected = []
    modulus = 1
    for p, g in available:
        if (p - 1) % size:
            continue
        selected.append((p, g))
        modulus *= p
        if modulus > 2 * bound:
            break
    if modulus <= 2 * bound:
        raise ValueError(
            "NTT modulus product insufficient for exact signed reconstruction"
        )
    out = [[0] * n for _ in range(m)]
    modulus = 1
    for p, g in selected:
        aa = [ntt([x % p for x in row] + [0] * (size - k), False, p, g) for row in a]
        bb = [
            ntt(
                [b[t][j] % p for t in range(k - 1, -1, -1)] + [0] * (size - k),
                False,
                p,
                g,
            )
            for j in range(n)
        ]
        inv = pow(modulus, -1, p)
        for i in range(m):
            for j in range(n):
                conv = ntt([x * y % p for x, y in zip(aa[i], bb[j])], True, p, g)
                out[i][j] += modulus * ((conv[k - 1] - out[i][j]) * inv % p)
        modulus *= p
    return [[x if x <= modulus // 2 else x - modulus for x in row] for row in out]


def finite_field_fp64(a, b):
    """Bounded FP64 GEMM modulo small primes, followed by exact Python CRT."""
    import numpy as np

    m, k, n = len(a), len(a[0]), len(b[0])
    out = [[0] * n for _ in range(m)]
    modulus = 1
    bound = k * max(abs(x) for r in a for x in r) * max(abs(x) for r in b for x in r)
    p = 101
    while True:
        while any(p % d == 0 for d in range(2, math.isqrt(p) + 1)):
            p += 1
        # Every partial dot (including backend reorderings) is below 2^52.
        block = max(1, min(k, ((1 << 52) - 1) // ((p - 1) * (p - 1))))
        residues = [[0] * n for _ in range(m)]
        for start in range(0, k, block):
            end = min(k, start + block)
            aa = np.array(
                [[x % p for x in row[start:end]] for row in a], dtype=np.float64
            )
            bb = np.array(
                [[x % p for x in row] for row in b[start:end]], dtype=np.float64
            )
            cc = aa @ bb
            for i in range(m):
                for j in range(n):
                    value = float(cc[i, j])
                    if not value.is_integer():
                        raise ArithmeticError(
                            "nonintegral FP64 finite-field intermediate"
                        )
                    residues[i][j] = (residues[i][j] + int(value)) % p
        inv = pow(modulus, -1, p)
        for i in range(m):
            for j in range(n):
                out[i][j] += modulus * ((residues[i][j] - out[i][j]) * inv % p)
        modulus *= p
        if modulus > 2 * bound:
            return [
                [x if x <= modulus // 2 else x - modulus for x in row] for row in out
            ]
        p += 1


def morton(a, b, tile=4):
    """Physical Morton order of fixed row-major tiles, exact scalar leaves."""
    if tile < 1:
        raise ValueError("positive tile size required")
    m, k, n = len(a), len(a[0]), len(b[0])

    def order(i, j):
        value = 0
        bit = 0
        while i or j:
            value |= (i & 1) << (2 * bit)
            value |= (j & 1) << (2 * bit + 1)
            i >>= 1
            j >>= 1
            bit += 1
        return value

    def pack(x):
        rows, cols = len(x), len(x[0])
        data = []
        indices = {}
        tiles = sorted(
            (
                (i, j)
                for i in range((rows + tile - 1) // tile)
                for j in range((cols + tile - 1) // tile)
            ),
            key=lambda q: order(*q),
        )
        for ti, tj in tiles:
            indices[ti, tj] = len(data)
            data.extend(
                (
                    x[ti * tile + i][tj * tile + j]
                    if ti * tile + i < rows and tj * tile + j < cols
                    else 0
                )
                for i in range(tile)
                for j in range(tile)
            )
        return data, indices

    aa, ai = pack(a)
    bb, bi = pack(b)
    out = [[0] * n for _ in range(m)]
    for (ti, tj), dummy in sorted(
        (
            (q, order(*q))
            for q in (
                (i, j)
                for i in range((m + tile - 1) // tile)
                for j in range((n + tile - 1) // tile)
            )
        ),
        key=lambda q: q[1],
    ):
        for tk in range((k + tile - 1) // tile):
            xa, xb = ai[ti, tk], bi[tk, tj]
            for i in range(min(tile, m - ti * tile)):
                for t in range(tile):
                    x = aa[xa + i * tile + t]
                    for j in range(min(tile, n - tj * tile)):
                        out[ti * tile + i][tj * tile + j] += x * bb[xb + t * tile + j]
    return out
