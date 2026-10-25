"""Create a reproducible text fixture. This never reuses an output computation."""

import argparse
import json
from pathlib import Path
import random
import hashlib

p = argparse.ArgumentParser()
p.add_argument("output", type=Path)
p.add_argument("--m", type=int, default=31)
p.add_argument("--k", type=int, default=67)
p.add_argument("--n", type=int, default=29)
p.add_argument("--seed", type=int, default=1234567)
args = p.parse_args()
if min(args.m, args.k, args.n) < 1:
    p.error("positive dimensions required")
rng = random.Random(args.seed)
with args.output.open("x") as f:
    for marker, rows, cols in (
        ("#A", args.m, args.k),
        ("#B", args.k, args.n),
        ("#C", args.m, args.n),
    ):
        f.write(marker + "\n")
        for _ in range(rows):
            f.write(" ".join(str(rng.randrange(-99, 100)) for _ in range(cols)) + "\n")
print(
    json.dumps(
        {
            "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "shape": [args.m, args.k, args.n],
            "seed": args.seed,
        }
    )
)
