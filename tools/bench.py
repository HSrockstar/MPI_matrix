"""Randomized paired end-to-end A/B benchmark with exact output verification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from probe import collect
from launch import run as run_job

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from exact import read_text, naive, add, write_text


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def summary(values):
    ordered = sorted(values)
    quantiles = (
        statistics.quantiles(ordered, n=4, method="inclusive")
        if len(values) > 1
        else [values[0]] * 3
    )
    return {
        "n": len(values),
        "median_s": statistics.median(values),
        "q1_s": quantiles[0],
        "q3_s": quantiles[2],
        "min_s": min(values),
        "max_s": max(values),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "mpi_matmul")
    p.add_argument("--np", type=int, default=4)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--expected", type=Path)
    p.add_argument("--oracle", choices=["numpy", "python"], default="numpy")
    p.add_argument("--cores")
    p.add_argument("--seed", type=int, default=1234567)
    args = p.parse_args()
    if args.np < 1 or args.trials < 1 or args.warmup < 0 or args.timeout <= 0:
        p.error("invalid run counts/timeout")
    cases = json.loads(args.cases.read_text())
    names = [c["name"] for c in cases]
    if not cases or len(set(names)) != len(names):
        p.error("need cases with unique names")
    for case in cases:
        if not isinstance(case.get("args", []), list) or not all(
            isinstance(x, str) for x in case.get("args", [])
        ):
            p.error("case args must be string arrays")
        case["binary"] = str(Path(case.get("binary", args.binary)).resolve())
        if not Path(case["binary"]).is_file():
            p.error(f"missing binary {case['binary']}")
        case["binary_sha256"] = sha(case["binary"])
    args.result.mkdir(parents=True, exist_ok=False)
    launcher = shlex.split(os.environ.get("MPIEXEC", "mpiexec"))
    rng = random.Random(args.seed)
    inp = args.input.resolve()
    a, b, c = read_text(inp)
    m, k, n = len(a), len(a[0]), len(b[0])
    # Prepare oracle before timed launches; every run still parses and computes.
    expected = args.result / "expected.txt"
    if args.expected:
        expected.write_bytes(args.expected.read_bytes())
    elif args.oracle == "python":
        write_text(expected, add(naive(a, b), c))
    else:
        import numpy as np

        bound = k * max(abs(x) for r in a for x in r) * max(
            abs(x) for r in b for x in r
        ) + max(abs(x) for r in c for x in r)
        if bound > (1 << 63) - 1:
            p.error("NumPy int64 oracle range exceeded; use Python oracle")
        write_text(
            expected,
            (
                np.array(a, dtype=np.int64) @ np.array(b, dtype=np.int64)
                + np.array(c, dtype=np.int64)
            ).tolist(),
        )
    del a, b, c
    gold = sha(expected)
    meta = {
        "input": str(inp),
        "input_sha256": sha(inp),
        "expected_sha256": gold,
        "cases": cases,
        "np": args.np,
        "trials": args.trials,
        "warmup": args.warmup,
        "seed": args.seed,
        "environment": collect(),
        "timer": "launcher start to exit; no fsync, warm filesystem cache",
    }
    (args.result / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    samples = {name: [] for name in names}
    raw = (args.result / "samples.jsonl").open("w")
    try:
        for round_ in range(-args.warmup, args.trials):
            order = list(cases)
            rng.shuffle(order)
            for case in order:
                ident = f'{round_+args.warmup:04d}-{names.index(case["name"]):03d}'
                out = args.result / (ident + ".txt")
                cmd = launcher + ["-n", str(args.np)]
                if args.cores:
                    cmd += [
                        sys.executable,
                        str(ROOT / "tools/bind_rank.py"),
                        "--cores",
                        args.cores,
                        "--",
                    ]
                cmd += [
                    case["binary"],
                    str(inp),
                    str(out),
                    "--m",
                    str(m),
                    "--k",
                    str(k),
                    "--n",
                    str(n),
                ] + case.get("args", [])
                env = os.environ.copy()
                env.update(
                    {
                        key: "1"
                        for key in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                        )
                    }
                )
                env.update(case.get("env", {}))
                start = time.perf_counter_ns()
                run = run_job(cmd, env=env, timeout=args.timeout)
                elapsed = (time.perf_counter_ns() - start) * 1e-9
                (args.result / (ident + ".stdout")).write_text(run.stdout)
                (args.result / (ident + ".stderr")).write_text(run.stderr)
                digest = sha(out) if out.exists() else None
                record = {
                    "round": round_,
                    "case": case["name"],
                    "wall_s": elapsed,
                    "returncode": run.returncode,
                    "output_sha256": digest,
                    "command": cmd,
                }
                raw.write(json.dumps(record) + "\n")
                raw.flush()
                if run.returncode or digest != gold:
                    raise RuntimeError(
                        f"failed exact validation: {ident}; logs retained"
                    )
                out.unlink()
                if round_ >= 0:
                    samples[case["name"]].append(elapsed)
    finally:
        raw.close()
    if sha(inp) != meta["input_sha256"]:
        raise RuntimeError("input changed during benchmark")
    for case in cases:
        if sha(case["binary"]) != case["binary_sha256"]:
            raise RuntimeError("binary changed during benchmark")
    result = {name: summary(times) for name, times in samples.items()}
    # Pair by randomized round, not by independently sorted timing values.
    base = samples[names[0]]
    boot = random.Random(args.seed + 1)
    for name in names[1:]:
        ratios = [x / y for x, y in zip(base, samples[name])]
        medians = sorted(
            statistics.median(boot.choices(ratios, k=len(ratios))) for _ in range(2000)
        )
        result[name]["paired_speedup_median"] = statistics.median(ratios)
        result[name]["paired_speedup_bootstrap95"] = [medians[50], medians[1949]]
    (args.result / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
