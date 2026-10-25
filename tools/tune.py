"""Produce an explicit bounded candidate grid; bench.py verifies every launch."""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--np", type=int, default=4)
p.add_argument("--trials", type=int, default=30)
p.add_argument("--run", action="store_true")
p.add_argument("--cores")
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
cases = [{"name": "baseline", "args": []}]
for (mr, nr), kb, groups in itertools.product(
    ((4, 16), (6, 16), (8, 8), (4, 24)), (256, 512, 1024, 2048), (8, 16, 32)
):
    cases.append(
        {
            "name": f"mr{mr}-nr{nr}-kb{kb}-rows{mr*groups}",
            "args": [
                "--mr",
                str(mr),
                "--nr",
                str(nr),
                "--kb",
                str(kb),
                "--rows",
                str(mr * groups),
            ],
        }
    )
path = args.output / "candidates.json"
path.write_text(json.dumps(cases, indent=2) + "\n")
if args.run:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench.py")),
        args.input,
        "--cases",
        str(path),
        "--result",
        str(args.output / "runs"),
        "--np",
        str(args.np),
        "--trials",
        str(args.trials),
    ]
    if args.cores:
        cmd += ["--cores", args.cores]
    subprocess.run(cmd, check=True)
    summary = json.loads((args.output / "runs/summary.json").read_text())
    best = min(summary, key=lambda key: summary[key]["median_s"])
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "screening_winner": best,
                "case": next(x for x in cases if x["name"] == best),
                "requires_fresh_confirmation": True,
            },
            indent=2,
        )
        + "\n"
    )
print(path)
