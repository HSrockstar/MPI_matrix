"""Exec one MPI rank on one explicitly selected physical core, without root."""

import argparse
import json
import os
from pathlib import Path
import sys
from probe import topology

p = argparse.ArgumentParser()
p.add_argument("--cores", required=True, help="comma-separated CPUs, or auto")
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
if args.command and args.command[0] == "--":
    args.command = args.command[1:]
if not args.command:
    p.error("missing executable")
local = next(
    (
        int(os.environ[k])
        for k in (
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "PMI_LOCAL_RANK",
            "SLURM_LOCALID",
        )
        if k in os.environ
    ),
    None,
)
if local is None:
    p.error("MPI local rank is unknown; use a supported launcher")
items = topology()
bycpu = {x["cpu"]: x for x in items}
seen = set()
if args.cores == "auto":
    cores = []
    for entry in items:
        key = entry["socket"], entry["core"]
        if None in key:
            p.error("physical topology unavailable; cannot verify binding")
        if key not in seen:
            cores.append(entry["cpu"])
            seen.add(key)
else:
    cores = [int(x) for x in args.cores.split(",")]
seen = set()
for cpu in cores:
    if cpu not in bycpu:
        p.error(f"CPU {cpu} is outside this rank allowed affinity")
    entry = bycpu[cpu]
    key = entry["socket"], entry["core"]
    if None in key or key in seen:
        p.error("CPU list contains unknown or duplicate physical cores")
    seen.add(key)
if local >= len(cores):
    p.error("not enough physical cores for local MPI ranks")
cpu = cores[local]
os.sched_setaffinity(0, {cpu})
print(
    json.dumps({"binding": bycpu[cpu], "local_rank": local, "pid": os.getpid()}),
    file=sys.stderr,
    flush=True,
)
os.execvp(args.command[0], args.command)
