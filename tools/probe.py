"""Read-only environment/provenance and MPI collective configuration probe."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(args):
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {
            "argv": args,
            "returncode": run.returncode,
            "stdout": run.stdout,
            "stderr": run.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "unavailable": str(exc)}


def topology():
    allowed = sorted(os.sched_getaffinity(0))
    out = []
    for cpu in allowed:
        base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")

        def read(name):
            path = base / name
            return path.read_text().strip() if path.exists() else None

        out.append(
            {
                "cpu": cpu,
                "socket": read("physical_package_id"),
                "core": read("core_id"),
                "siblings": read("thread_siblings_list"),
                "core_type": read("core_type"),
            }
        )
    return out


def collect():
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MPIEXEC",
        "MPIR_CVAR_CH4_NETMOD",
        "FI_PROVIDER",
        "MATMUL_BLAS_LIBRARY",
        "MATMUL_BACKEND_LIBRARY",
        "OMPI_MCA_coll_tuned_use_dynamic_rules",
    )
    cg = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            cg[name] = path.read_text().strip()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "topology": topology(),
        "cgroup": cg,
        "env": {key: os.environ[key] for key in keys if key in os.environ},
        "tools": [
            command(args)
            for args in (
                ["lscpu", "-J"],
                ["mpiexec", "--version"],
                ["mpicc", "-show"],
                ["gcc", "--version"],
                ["clang", "--version"],
            )
            if shutil.which(args[0])
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--collectives", action="store_true")
    args = p.parse_args()
    data = collect()
    if args.collectives:
        # Query the installed version; do not assume numeric algorithm IDs.
        data["collectives"] = command(
            ["ompi_info", "--param", "coll", "all", "--level", "9"]
        )
    Path(args.output).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
