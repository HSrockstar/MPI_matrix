"""Input/output boundary contracts, including byte-level parser fuzz and int64 output."""

import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from exact import add, naive, write_text

launcher = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
binary = Path(os.environ.get("MATMUL_BINARY", str(ROOT / "mpi_matmul"))).resolve()
rng = random.Random(999)
cases = 0
with tempfile.TemporaryDirectory(prefix="mpi-matrix-boundary-") as td:
    td = Path(td)
    inp = td / "in"
    out = td / "out"
    gold = td / "gold"

    def run(a, b, c, flags, np_=4, malform=None, valid=True):
        global cases
        m, k, n = len(a), len(a[0]), len(b[0])
        pieces = []
        for marker, mat in zip(("#A", "#B", "#C"), (a, b, c)):
            pieces.append(marker + "\r\n")
            for row in mat:
                pieces.append(
                    "".join(
                        str(x)
                        + (
                            rng.choice((" ", "\t", "  "))
                            if i + 1 < len(row)
                            else "\r\n"
                        )
                        for i, x in enumerate(row)
                    )
                )
        text = "".join(pieces).rstrip("\r\n")  # Deliberately missing the final newline.
        if malform:
            text = malform(text)
        inp.write_text(text)
        out.unlink(missing_ok=True)
        write_text(gold, add(naive(a, b), c))
        cmd = (
            launcher
            + [
                "-n",
                str(np_),
                str(binary),
                str(inp),
                str(out),
                "--m",
                str(m),
                "--k",
                str(k),
                "--n",
                str(n),
            ]
            + flags
        )
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        if valid:
            if result.returncode or out.read_bytes() != gold.read_bytes():
                raise AssertionError(str(cmd) + "\n" + result.stdout + result.stderr)
        elif result.returncode == 0:
            raise AssertionError("invalid input was accepted")
        cases += 1

    for seed in range(20):
        m, k, n = rng.randrange(1, 12), rng.randrange(1, 85), rng.randrange(1, 25)
        a = [[rng.randrange(-99, 100) for _ in range(k)] for _ in range(m)]
        b = [[rng.randrange(-99, 100) for _ in range(n)] for _ in range(k)]
        c = [[rng.randrange(-99, 100) for _ in range(n)] for _ in range(m)]
        for parser in ("scalar", "swar", "simd", "wide"):
            run(
                a,
                b,
                c,
                [
                    "--parser",
                    parser,
                    "--arena",
                    "1",
                    "--balance",
                    "1000",
                    "--kb",
                    "3",
                    "--format",
                    "vector",
                ],
                1 + seed % 4,
            )
    for dist in ("rows", "ring", "split-k", "summa", "columns"):
        for kernel in ("auto", "recursive", "strassen", "strassen-winograd"):
            run(
                [[99] * 2047],
                [[-99] for _ in range(2047)],
                [[0]],
                ["--dist", dist, "--kernel", kernel, "--kb", "513"],
                4,
            )
    a = [[-2147483648], [2147483647]]
    b = [[1, -1, 2147483647]]
    c = [[0, 0, 0], [0, 0, 0]]
    for fmt in ("pairs", "quad", "jeaiii", "snprintf", "batch", "vector"):
        for dist in ("rows", "split-k", "summa", "ring"):
            run(
                a,
                b,
                c,
                [
                    "--kernel",
                    "scalar",
                    "--dist",
                    dist,
                    "--format",
                    fmt,
                    "--parser",
                    "wide",
                ],
            )
    for parser in ("scalar", "swar", "simd", "wide"):
        for change in (
            lambda s: s.replace("#B", "#A"),
            lambda s: s.replace("99", "99x", 1),
            lambda s: s.replace("99", "+", 1),
            lambda s: s.replace("99", "--9", 1),
            lambda s: s + " 1",
            lambda s: s.replace("99", "", 1),
            lambda s: s.replace("99", "2147483648", 1),
        ):
            run(
                [[99, 99]],
                [[99], [99]],
                [[0]],
                ["--parser", parser],
                malform=change,
                valid=False,
            )
    run(
        [[-2147483648, -2147483648]],
        [[-2147483648], [-2147483648]],
        [[0]],
        ["--kernel", "scalar"],
        valid=False,
    )
    # A zero-size rank must participate in collective output, including MPI-IO.
    for writer in ("pwrite", "mpi", "mpi-all"):
        run(
            [[99]],
            [[99]],
            [[99]],
            [
                "--writer",
                writer,
                "--stream",
                "1",
                "--fused",
                "1",
                "--bmode",
                "root-packed",
                "--balance",
                "1000",
                "--rows",
                "6",
            ],
        )
print(json.dumps({"boundary_cases_passed": cases}))
