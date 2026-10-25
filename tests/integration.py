"""Deterministic end-to-end tests. MPIEXEC may include launcher options."""

import argparse
import itertools
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
from exact import naive, add, write_text


def fixture(path, m, k, n, seed=1234567, crlf=False, extreme=False):
    rng = random.Random(seed)
    a = [
        [99 if extreme else rng.randrange(-99, 100) for _ in range(k)] for _ in range(m)
    ]
    b = [
        [-99 if extreme else rng.randrange(-99, 100) for _ in range(n)]
        for _ in range(k)
    ]
    c = [[rng.randrange(-99, 100) for _ in range(n)] for _ in range(m)]
    text = "".join(
        marker + "\n" + "".join(" ".join(map(str, row)) + "\n" for row in matrix)
        for marker, matrix in zip(("#A", "#B", "#C"), (a, b, c))
    )
    path.write_bytes(text.replace("\n", "\r\n").encode() if crlf else text.encode())
    return add(naive(a, b), c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--models", action="store_true")
    args = p.parse_args()
    launcher = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
    binary = Path(os.environ.get("MATMUL_BINARY", str(ROOT / "mpi_matmul"))).resolve()
    features_run = subprocess.run(
        launcher + ["-n", "1", str(binary), "--features"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    features = json.loads(features_run.stdout.strip())
    if not features["avx2"]:
        raise RuntimeError(
            "full experiment suite requires AVX2; use --selftest for scalar-only CPUs"
        )
    simd_kernel = "vnni" if features["avxvnni"] else "avx2-sign"
    tests = 0
    with tempfile.TemporaryDirectory(prefix="mpi-matrix-test-") as td:
        td = Path(td)
        inp = td / "input.txt"
        out = td / "output.txt"
        expected = td / "expected.txt"
        m, k, n = 7, 19, 17
        write_text(expected, fixture(inp, m, k, n))
        gold = expected.read_bytes()
        configs = []
        for np in (1, 2, 3, 4):
            for dist in ("rows", "ring", "split-k", "summa", "columns"):
                configs.append((np, ["--dist", dist, "--kb", "7"]))
            for mode in ("allgatherv", "equal", "root", "replicate", "packed"):
                configs.append((np, ["--bmode", mode, "--kb", "5"]))
        if not args.quick:
            for kernel in (
                "scalar",
                "recursive",
                "fp32",
                "fp64",
                "winograd",
                "vnni",
                "avx2-sign",
                "nibble",
                "blas",
                "blas-fp32",
                "strassen",
                "strassen-winograd",
            ):
                configs.append((3, ["--kernel", kernel, "--kb", "3"]))
            for mr, nr in ((4, 16), (6, 16), (8, 8), (4, 24)):
                for layout in ("rows", "interleaved"):
                    configs.append(
                        (
                            4,
                            [
                                "--mr",
                                str(mr),
                                "--nr",
                                str(nr),
                                "--rows",
                                str(mr * 3),
                                "--layout",
                                layout,
                                "--pad",
                                "64",
                                "--unroll",
                                "2",
                            ],
                        )
                    )
            for key, values in {
                "parser": ["scalar", "simd", "swar", "wide"],
                "reader": ["mmap", "pread"],
                "prefault": ["none", "local", "all"],
                "format": ["pairs", "quad", "jeaiii", "snprintf", "batch", "vector"],
                "writer": ["pwrite", "mpi", "mpi-all"],
                "offset": ["allgather", "exscan"],
                "alloc": ["aligned", "huge"],
                "arena": ["0", "1"],
                "beta0": ["0", "1"],
                "progress": ["0", "1"],
                "pipeline": ["0", "1"],
                "aligned": ["0", "1"],
                "progress-us": ["50", "100", "200"],
                "memory-stats": ["1"],
                "write-chunk": ["1", "31"],
                "balance": ["1", "1000"],
            }.items():
                for value in values:
                    configs.append((4, ["--" + key, value, "--kb", "5"]))
            configs.extend(
                [
                    (4, ["--stream", "1", "--bmode", "root", "--rows", "6"]),
                    (
                        3,
                        [
                            "--stream",
                            "1",
                            "--bmode",
                            "replicate",
                            "--rows",
                            "6",
                            "--kernel",
                            "avx2-sign",
                        ],
                    ),
                    (4, ["--format", "quad", "--static-table", "1"]),
                ]
            )
        for kern in ("vnni", "avx2-sign", "nibble"):
            for layout in ("rows", "interleaved"):
                for mode in ("root", "root-packed", "replicate"):
                    configs.append(
                        (
                            4,
                            [
                                "--stream",
                                "1",
                                "--fused",
                                "1",
                                "--bmode",
                                mode,
                                "--kernel",
                                kern,
                                "--layout",
                                layout,
                                "--arena",
                                "1",
                                "--parser",
                                "wide",
                                "--rows",
                                "6",
                                "--pipeline",
                                "1",
                            ],
                        )
                    )
        if not args.quick:
            for mr, nr in ((4, 16), (6, 16), (8, 8), (4, 24)):
                for kernel in ("vnni", "avx2-sign", "nibble"):
                    for unroll, pipeline in ((1, 0), (2, 0), (4, 0), (1, 1)):
                        configs.append(
                            (
                                3,
                                [
                                    "--kernel",
                                    kernel,
                                    "--mr",
                                    str(mr),
                                    "--nr",
                                    str(nr),
                                    "--rows",
                                    str(mr * 2),
                                    "--unroll",
                                    str(unroll),
                                    "--pipeline",
                                    str(pipeline),
                                    "--aligned",
                                    "1",
                                    "--bmode",
                                    "packed",
                                    "--kb",
                                    "7",
                                ],
                            )
                        )
        for np, flags in configs:
            flags = [simd_kernel if x == "vnni" else x for x in flags]
            out.unlink(missing_ok=True)
            cmd = (
                launcher
                + [
                    "-n",
                    str(np),
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
            run = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if run.returncode or not out.exists() or out.read_bytes() != gold:
                raise AssertionError(f"FAILED {cmd}\n{run.stdout}\n{run.stderr}")
            metrics = [
                json.loads(line)
                for line in run.stdout.splitlines()
                if line.startswith("{")
            ]
            assert metrics and metrics[-1]["np"] == np
            tests += 1
        # Empty row/K owners, odd tails, extreme values, CRLF and no final LF.
        for m, k, n in ((1, 1, 1), (2, 3, 5), (9, 2047, 19)):
            goldpath = td / "edge.txt"
            write_text(goldpath, fixture(inp, m, k, n, crlf=True, extreme=True))
            gold = goldpath.read_bytes()
            for dist in ("rows", "split-k", "summa", "ring"):
                for kernel in ("vnni", "avx2-sign", "fp32"):
                    flags = [
                        "--dist",
                        dist,
                        "--kernel",
                        simd_kernel if kernel == "vnni" else kernel,
                        "--kb",
                        "513",
                        "--parser",
                        "simd",
                    ]
                    cmd = (
                        launcher
                        + [
                            "-n",
                            "4",
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
                    run = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=60
                    )
                    if run.returncode or out.read_bytes() != gold:
                        raise AssertionError(f"edge {cmd}\n{run.stdout}\n{run.stderr}")
                    tests += 1
        # Reject malformed inputs, overflowing fallback and invalid options.
        original = inp.read_bytes()
        for data, extra in (
            (original.replace(b"#A", b"#X", 1), []),
            (original.replace(b"99", b"999999999999999999", 1), []),
            (original, ["--mr", "5"]),
            (original, ["--kernel", "unknown"]),
            (original.replace(b"99", b"100", 1), ["--kernel", "vnni"]),
        ):
            inp.write_bytes(data)
            run = subprocess.run(
                launcher
                + [
                    "-n",
                    "2",
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
                + extra,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if run.returncode == 0:
                raise AssertionError("invalid case accepted")
            tests += 1
        inp.write_bytes(original)
        run = subprocess.run(
            launcher
            + [
                "-n",
                "2",
                str(binary),
                str(inp),
                str(inp),
                "--m",
                str(m),
                "--k",
                str(k),
                "--n",
                str(n),
            ],
            capture_output=True,
            timeout=30,
        )
        assert run.returncode and inp.read_bytes() == original
        tests += 1
        if args.models:
            m, k, n = 7, 11, 9
            write_text(expected, fixture(inp, m, k, n))
            gold = expected.read_bytes()
            for alg in (
                "cannon",
                "fox",
                "layered-summa",
                "hsumma",
                "recursive",
                "grid-search",
            ):
                flags = ["--algorithm", alg, "--kb", "3"]
                if alg == "layered-summa":
                    flags += ["--layers", "2"]
                run = subprocess.run(
                    launcher
                    + [
                        "-n",
                        "4",
                        sys.executable,
                        str(ROOT / "lab/mpi_models.py"),
                        str(inp),
                        str(out),
                    ]
                    + flags,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if run.returncode or out.read_bytes() != gold:
                    raise AssertionError(f"model {alg}\n{run.stdout}\n{run.stderr}")
                tests += 1
            for np in (1, 2, 3, 4):
                run = subprocess.run(
                    launcher
                    + [
                        "-n",
                        str(np),
                        sys.executable,
                        str(ROOT / "lab/fused_parse.py"),
                        str(inp),
                        str(out),
                        "--m",
                        str(m),
                        "--k",
                        str(k),
                        "--n",
                        str(n),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if run.returncode or out.read_bytes() != gold:
                    raise AssertionError(
                        f"fused parse {np}\n{run.stdout}\n{run.stderr}"
                    )
                tests += 1
    print(json.dumps({"integration_cases_passed": tests, "models": args.models}))


if __name__ == "__main__":
    main()
