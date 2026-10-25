"""Contract test for any compiled optional adapter; never counts missing libs as pass."""

import argparse
import ctypes as C
import json
from pathlib import Path
import random

p = argparse.ArgumentParser()
p.add_argument("library", type=Path)
args = p.parse_args()
lib = C.CDLL(str(args.library.resolve()))
fn = lib.matrix_backend_v1
fn.argtypes = (
    [C.POINTER(C.c_int32), C.POINTER(C.c_int32), C.POINTER(C.c_int64)]
    + [C.c_int] * 7
    + [C.c_char_p, C.c_size_t]
)
fn.restype = C.c_int
rng = random.Random(1234567)
tests = 0
for m, k, n in (
    (0, 3, 5),
    (1, 0, 1),
    (1, 1, 1),
    (5, 7, 9),
    (7, 513, 19),
    (9, 2047, 17),
):
    la, lb, ld = k + 3, n + 5, n + 7
    for value in (99, -99, None):
        aa = [
            rng.randrange(-99, 100) if value is None else value for _ in range(m * la)
        ]
        bb = [rng.randrange(-99, 100) if value is None else 99 for _ in range(k * lb)]
        a = (C.c_int32 * len(aa))(*aa)
        b = (C.c_int32 * len(bb))(*bb)
        for add in (0, 1):
            d = (C.c_int64 * (m * ld))(*([1234567] * (m * ld)))
            error = C.create_string_buffer(256)
            rc = fn(a, b, d, m, k, n, la, lb, ld, add, error, 256)
            if rc:
                raise AssertionError(error.value.decode())
            for i in range(m):
                for j in range(n):
                    assert d[i * ld + j] == sum(
                        aa[i * la + t] * bb[t * lb + j] for t in range(k)
                    ) + (1234567 if add else 0), (m, k, n, i, j)
                assert list(d)[i * ld + n : (i + 1) * ld] == [1234567] * (ld - n)
            tests += 1
print(json.dumps({"adapter_cases_passed": tests, "library": str(args.library)}))
