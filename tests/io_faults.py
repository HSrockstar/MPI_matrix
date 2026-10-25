"""Fault-inject POSIX partial writes/EINTR/zero progress/ENOSPC."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
sys.path.insert(0, str(ROOT / "tools"))
from exact import naive, add, write_text
from launch import run

launcher = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
with tempfile.TemporaryDirectory(prefix="mpi-matrix-io-") as td:
    td = Path(td)
    plugin = td / "fault.so"
    inp = td / "in"
    out = td / "out"
    gold = td / "gold"
    subprocess.run(
        [
            "cc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fPIC",
            "-shared",
            str(ROOT / "tests/io_fault.c"),
            "-ldl",
            "-o",
            str(plugin),
        ],
        check=True,
    )
    a = [[i * 3 + t - 40 for t in range(19)] for i in range(7)]
    b = [[t - j for j in range(17)] for t in range(19)]
    c = [[0] * 17 for _ in range(7)]
    inp.write_text(
        "".join(
            marker + "\n" + "".join(" ".join(map(str, row)) + "\n" for row in matrix)
            for marker, matrix in zip(("#A", "#B", "#C"), (a, b, c))
        )
    )
    write_text(gold, add(naive(a, b), c))
    for mode in ("short-eintr", "zero", "error"):
        out.unlink(missing_ok=True)
        env = os.environ.copy()
        env.update({"LD_PRELOAD": str(plugin), "MATRIX_IO_FAULT": mode})
        result = run(
            launcher
            + [
                "-n",
                "4",
                str(ROOT / "mpi_matmul"),
                str(inp),
                str(out),
                "--m",
                "7",
                "--k",
                "19",
                "--n",
                "17",
            ],
            env=env,
            timeout=30,
        )
        if mode == "short-eintr":
            assert result.returncode == 0, result.stdout + result.stderr
            assert out.read_bytes() == gold.read_bytes()
        else:
            assert result.returncode != 0, "write error was ignored"
print(json.dumps({"io_fault_cases_passed": 3}))
