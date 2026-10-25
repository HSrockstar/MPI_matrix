"""Full assignment dimensions, generated inputs and independent int64 oracle."""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import numpy as np

root = Path(__file__).resolve().parents[1]
launcher = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
with tempfile.TemporaryDirectory(prefix="mpi-matrix-full-") as td:
    td = Path(td)
    rng = np.random.default_rng(1234567)
    m, k, n = 1023, 2047, 1023
    a = rng.integers(-99, 100, size=(m, k), dtype=np.int64)
    b = rng.integers(-99, 100, size=(k, n), dtype=np.int64)
    c = rng.integers(-99, 100, size=(m, n), dtype=np.int64)
    start = time.perf_counter()
    reference = a @ b + c
    with (td / "input.txt").open("w") as f:
        for marker, mat in zip(("#A", "#B", "#C"), (a, b, c)):
            f.write(marker + "\n")
            np.savetxt(f, mat, fmt="%d")
    print(
        json.dumps(
            {
                "oracle_and_fixture_s": time.perf_counter() - start,
                "input_sha256": hashlib.sha256(
                    (td / "input.txt").read_bytes()
                ).hexdigest(),
            }
        ),
        flush=True,
    )
    cases = [(p, ["--kernel", "vnni", "--parser", "simd"]) for p in (1, 2, 4)]
    cases += [
        (
            4,
            [
                "--kernel",
                "avx2-sign",
                "--bmode",
                "packed",
                "--kb",
                "513",
                "--format",
                "vector",
            ],
        ),
        (3, ["--dist", "split-k", "--kernel", "vnni"]),
        (
            4,
            [
                "--stream",
                "1",
                "--fused",
                "1",
                "--bmode",
                "root-packed",
                "--parser",
                "wide",
                "--format",
                "jeaiii",
            ],
        ),
        (4, ["--dist", "summa", "--kernel", "blas-fp32", "--kb", "1024"]),
    ]
    cases += [
        (4, ["--kernel", "strassen", "--parser", "wide"]),
        (4, ["--kernel", "strassen-winograd", "--parser", "wide"]),
    ]
    for np_, flags in cases:
        out = td / "out.txt"
        out.unlink(missing_ok=True)
        start = time.perf_counter()
        run = subprocess.run(
            launcher
            + [
                "-n",
                str(np_),
                str(root / "mpi_matmul"),
                str(td / "input.txt"),
                str(out),
            ]
            + flags,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if run.returncode:
            raise AssertionError(run.stdout + run.stderr)
        actual = np.loadtxt(out, skiprows=1, dtype=np.int64)
        assert np.array_equal(actual, reference), (np_, flags)
        print(
            json.dumps(
                {
                    "np": np_,
                    "flags": flags,
                    "validation_wall_s": time.perf_counter() - start,
                    "exact": True,
                }
            ),
            flush=True,
        )
