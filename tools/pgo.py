"""GCC profile generate/train/use with identical object paths and strict diagnostics."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("input", type=Path)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--mpicc", default="mpicc")
p.add_argument("--mpicxx", default="mpicxx")
args = p.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=False)
# Unique object directory is shared by both phases and never touches release objects.
build = args.output / "objects"
profile = args.output / "profile"
profile.mkdir()
binary = args.output / "mpi_matmul_pgo"
base = "-O3 -g -Wall -Wextra -fprofile-update=atomic"
common = [
    "make",
    "-C",
    str(ROOT),
    "-B",
    "-j2",
    f"BUILD={build}",
    f"TARGET={binary}",
    f"MPICC={args.mpicc}",
    f"MPICXX={args.mpicxx}",
]


def compile_(flag, phase):
    cmd = common + [
        f"CFLAGS={base} -std=c11 {flag}",
        f"CXXFLAGS={base} -std=c++11 {flag}",
        f"LDFLAGS={flag}",
    ]
    with (args.output / (phase + ".log")).open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


compile_(f"-fprofile-generate={profile}", "generate")
cases = args.output / "cases.json"
cases.write_text(
    json.dumps([{"name": "pgo", "binary": str(binary), "args": ["--parser", "wide"]}])
)
for np_ in (1, 2, 4):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/bench.py"),
            str(args.input.resolve()),
            "--cases",
            str(cases),
            "--result",
            str(args.output / f"train-{np_}"),
            "--np",
            str(np_),
            "--trials",
            "1",
            "--warmup",
            "0",
        ],
        check=True,
    )
if not list(profile.rglob("*.gcda")):
    raise RuntimeError("no GCC profile data produced")
compile_(
    f"-fprofile-use={profile} -fprofile-correction -Werror=missing-profile -Werror=coverage-mismatch",
    "use",
)
subprocess.run(
    [
        sys.executable,
        str(ROOT / "tools/bench.py"),
        str(args.input.resolve()),
        "--cases",
        str(cases),
        "--result",
        str(args.output / "verify-use"),
        "--np",
        "4",
        "--trials",
        "1",
        "--warmup",
        "0",
    ],
    check=True,
)
print(binary)
